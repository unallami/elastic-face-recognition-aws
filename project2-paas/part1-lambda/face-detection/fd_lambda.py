import os
import io
import json
import base64
import logging
import traceback
from typing import Optional

import boto3
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

REQUEST_QUEUE_URL = os.environ.get("REQUEST_QUEUE_URL")  
AWS_REGION = "us-east-1"
FACE_CROP_SIZE = int(os.environ.get("FACE_CROP_SIZE", "160"))
MARGIN = int(os.environ.get("MARGIN", "0"))
MAX_MESSAGE_BYTES = int(os.environ.get("MAX_MESSAGE_BYTES", "1000"))

torch.set_num_threads(1)

logger = logging.getLogger("face_detection")
logger.setLevel(logging.INFO)


device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(
    image_size=FACE_CROP_SIZE,
    margin=MARGIN,
    keep_all=True,
    post_process=True,
    device=device,
)

sqs = boto3.client("sqs", region_name=AWS_REGION)


def _b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _tensor_to_pil(face_tensor) -> Image.Image:
    t = face_tensor
    if getattr(t, "ndim", 0) != 3:
        t = t.squeeze(0)
    t = t.clamp(0, 1)
    np_img = (t.mul(255).byte().permute(1, 2, 0).cpu().numpy())
    if np_img.shape[2] == 1:
        np_img = np.repeat(np_img, 3, axis=2)
    return Image.fromarray(np_img, mode="RGB")


def _jpeg_b64(img: Image.Image, quality: int, mode: str = "RGB") -> str:
    work = img.convert(mode)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _try_pack_face(request_id: str, face_img: Image.Image) -> Optional[str]:
    w, h = face_img.size
    base = max(w, h)
    sizes = []
    curr = base
    while curr > 64:
        sizes.append(curr)
        curr = int(curr * 0.85)
    sizes.append(64)

    qualities = [80, 70, 60, 50, 45, 40, 35]
    modes = ["RGB", "L"]

    for mode in modes:
        for s in sizes:
            scaled = face_img if face_img.size == (s, s) else face_img.resize((s, s), Image.BILINEAR)
            for q in qualities:
                b64 = _jpeg_b64(scaled, q, mode=mode)
                payload = {"request_id": request_id, "face": b64}
                msg = json.dumps(payload, separators=(",", ":"))
                if len(msg.encode("utf-8")) < MAX_MESSAGE_BYTES:
                    return b64
    return None


def _ok(body_dict):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict),
    }


def _err(msg, exc: Exception = None):
    if exc:
        logger.exception(msg)
    else:
        logger.error(msg)
    return {
        "statusCode": 500,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": msg}),
    }


def lambda_handler(event, context):
    try:
        if isinstance(event, dict) and "body" in event:
            body_raw = event.get("body")
            data = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
        elif isinstance(event, dict):
            data = event
        else:
            return _err("invalid request body")

        request_id = data.get("request_id", "")
        content_b64 = data.get("content") or data.get("image") or data.get("content_b64")
        filename = data.get("filename", "")

        if not (request_id and content_b64):
            return _err("request_id and content are required")

        if not REQUEST_QUEUE_URL:
            return _err("REQUEST_QUEUE_URL env var is not set")

        img = _b64_to_pil(content_b64)

        faces, probs = mtcnn(img, return_prob=True)
        if faces is None or (hasattr(faces, "shape") and faces.shape[0] == 0):
            return _ok({"request_id": request_id, "filename": filename, "faces_detected": 0, "faces_enqueued": 0})

        n = int(faces.shape[0]) if hasattr(faces, "shape") else 0
        sent = 0

        for i in range(n):
            try:
                face_pil = _tensor_to_pil(faces[i])
                b64 = _try_pack_face(request_id, face_pil)
                if not b64:
                    logger.warning(f"Skipping face {i}: cannot fit message under {MAX_MESSAGE_BYTES} bytes")
                    continue

                payload = {"request_id": request_id, "face": b64}
                msg = json.dumps(payload, separators=(",", ":"))
                if len(msg.encode("utf-8")) >= MAX_MESSAGE_BYTES:
                    logger.warning(f"Skipping face {i}: final message still >= {MAX_MESSAGE_BYTES} bytes")
                    continue

                try:
                    sqs.send_message(QueueUrl=REQUEST_QUEUE_URL, MessageBody=msg)
                    sent += 1
                except Exception as e:
                    logger.exception("Failed to send SQS message")
                    continue

            except Exception as e:
                logger.exception("Error processing face index %s", i)
                continue

        return _ok({
            "request_id": request_id,
            "filename": filename,
            "faces_detected": n,
            "faces_enqueued": sent
        })

    except Exception as e:
        return _err("Unhandled exception in face-detection", exc=e)
