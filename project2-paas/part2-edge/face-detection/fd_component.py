import sys
import io
import json
import base64
import logging
import time
import signal
import threading
from typing import Optional

import boto3
import torch
import numpy as np
from PIL import Image

import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import SubscribeToIoTCoreRequest, QOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQ_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/577327406267/1233809163-req-queue"
RESP_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/577327406267/1233809163-resp-queue"

sqs_client = boto3.client("sqs", region_name="us-east-1")
torch.set_num_threads(1)

stop_flag = threading.Event()
ipc_operation = None
TOPIC = "clients/1233809163-IoTThing"

def jpeg_b64_encode(img: Image.Image, quality: int, mode: str = "RGB") -> str:
    buf = io.BytesIO()
    img.convert(mode).save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_b64_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def tensor_to_pil_image(tensor) -> Image.Image:
    t = tensor
    is_torch = "torch" in str(type(t)).lower()
    if is_torch:
        t = t.detach().cpu().float()

    if getattr(t, "ndim", 0) >= 4:
        t = t[0] if (t.shape[0] > 1) else t.squeeze(0)

    if getattr(t, "ndim", 0) == 2:
        t = t.unsqueeze(0) if is_torch else np.expand_dims(t, 0)
    else:
        if getattr(t, "ndim", 0) not in (2, 3):
            t = t.squeeze()

    arr = t.cpu().numpy() if is_torch else np.array(t)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr)

    if arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)

    a_min, a_max = float(arr.min()), float(arr.max())
    if a_min >= 0 and a_max <= 1:
        norm = arr
    elif a_min >= 0 and a_max <= 255:
        norm = arr / 255.0
    else:
        span = a_max - a_min
        norm = (arr - a_min) / span if span > 1e-6 else np.clip(arr, 0, 1)

    norm = np.clip(norm, 0, 1)
    img = (norm * 255).astype(np.uint8)

    if img.shape[0] == 1:
        return Image.fromarray(img[0], "L").convert("RGB")
    else:
        return Image.fromarray(np.transpose(img, (1, 2, 0)), "RGB")


def pack_face_candidate(req_id: str, face_img: Image.Image) -> Optional[str]:
    # tries multiple sizes/qualities
    w, h = face_img.size
    base = max(w, h)
    sizes = []
    s = base
    while s > 64:
        sizes.append(s)
        s = int(s * 0.85)
    sizes.append(64)

    for mode in ["RGB", "L"]:
        for size in sizes:
            scaled = face_img if face_img.size == (size, size) else face_img.resize((size, size), Image.BILINEAR)
            for q in [80, 70, 60, 50, 45, 40, 35]:
                b64 = jpeg_b64_encode(scaled, q, mode)
                msg = json.dumps({"request_id": req_id, "face": b64}, separators=(",", ":"))
                if len(msg.encode("utf-8")) < 1000:
                    return b64
    return None

sys.path.insert(0, '/greengrass/v2/packages/artifacts/com.clientdevices.FaceDetection/1.0.0/facenet_pytorch')
from models.mtcnn import MTCNN

detector = MTCNN(
    image_size=160,
    margin=0,
    keep_all=True,
    post_process=True,
    device="cpu",
)

def handle_message(payload: dict):
    try:
        req_id = payload.get("request_id", "")
        raw_b64 = (
            payload.get("encoded")
            or payload.get("content")
            or payload.get("image")
            or payload.get("content_b64")
        )
        fname = payload.get("filename", "")

        if not req_id or not raw_b64:
            return {"error": "request_id and content are required"}

        img = decode_b64_image(raw_b64)
        try:
            faces, probs = detector(img, return_prob=True)
        except Exception:
            return {"request_id": req_id, "filename": fname, "faces_detected": 0, "faces_enqueued": 0}

        if faces is None or (hasattr(faces, "shape") and faces.shape[0] == 0):
            reply = {"request_id": req_id, "result": "No-Face"}
            sqs_client.send_message(QueueUrl=RESP_QUEUE_URL, MessageBody=json.dumps(reply))
            return {"request_id": req_id, "filename": fname, "faces_detected": 0, "faces_enqueued": 0}

        n = faces.shape[0]
        if probs is None:
            probs_arr = np.zeros((n,), dtype=float)
        else:
            try:
                probs_arr = np.array([float(p or 0) for p in probs], dtype=float)
            except Exception:
                probs_arr = np.asarray(probs, dtype=float)

        if probs_arr.size > 1:
            idx = int(np.nanargmax(probs_arr))
            faces = faces[idx:idx + 1]
            probs_arr = np.array([probs_arr[idx]])
            n = 1

        sent = 0
        for i in range(n):
            image_pil = tensor_to_pil_image(faces[i])
            b64_face = pack_face_candidate(req_id, image_pil)
            if not b64_face:
                continue
            msg = json.dumps({"request_id": req_id, "face": b64_face}, separators=(",", ":"))
            if len(msg.encode("utf-8")) >= 1000:
                continue

            sqs_client.send_message(QueueUrl=REQ_QUEUE_URL, MessageBody=msg)
            sent += 1

        return {
            "request_id": req_id,
            "filename": fname,
            "faces_detected": n,
            "faces_enqueued": sent,
        }

    except Exception:
        logger.exception("processing error")
        return {"error": "Unhandled exception"}


class IoTStreamHandler(awsiot.greengrasscoreipc.client.SubscribeToIoTCoreStreamHandler):
    def on_stream_event(self, event):
        try:
            candidates = []
            msgobj = getattr(event, "message", None) or getattr(event, "msg", None)

            if msgobj is not None:
                if getattr(msgobj, "payload", None) is not None:
                    candidates.append(msgobj.payload)
                if getattr(msgobj, "data", None) is not None:
                    candidates.append(msgobj.data)

            if getattr(event, "payload", None) is not None:
                candidates.append(event.payload)
            if getattr(event, "data", None) is not None:
                candidates.append(event.data)

            raw = None
            for c in candidates:
                if isinstance(c, (bytes, bytearray)):
                    raw = bytes(c)
                    break
                if hasattr(c, "tobytes"):
                    raw = c.tobytes()
                    break
                if isinstance(c, str):
                    raw = c.encode("utf-8")
                    break

            if raw is None:
                return

            text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(text)
            except Exception:
                return

            result = handle_message(data)
            logger.info("result=%s", result)

        except Exception:
            logger.exception("stream error")

    def on_stream_error(self, error):
        logger.exception("stream error: %s", error)

    def on_stream_closed(self):
        logger.info("stream closed")


def _signal_handler(signum, frame):
    stop_flag.set()


def run_ipc_listener():
    global ipc_operation
    client = awsiot.greengrasscoreipc.connect()
    handler = IoTStreamHandler()
    req = SubscribeToIoTCoreRequest(topic_name=TOPIC, qos=QOS.AT_LEAST_ONCE)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        ipc_operation = client.new_subscribe_to_iot_core(handler)
        ipc_operation.activate(req)
    except Exception:
        logger.exception("failed to start subscription")
        return

    while not stop_flag.is_set():
        time.sleep(1)

    try:
        ipc_operation.close()
    except Exception:
        pass

if __name__ == "__main__":
    run_ipc_listener()
