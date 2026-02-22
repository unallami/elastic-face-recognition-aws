import os

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import io
import json
import base64
import logging
import traceback
from typing import Tuple

import boto3
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

RESPONSE_QUEUE_URL = os.environ.get("RESPONSE_QUEUE_URL")  
AWS_REGION = "us-east-1"
FACEBANK_PATH = os.environ.get("FACEBANK_PATH", "/var/task/resnetV1_video_weights_1.pt")
USE_UNKNOWN_THRESHOLD = os.environ.get("USE_UNKNOWN_THRESHOLD", "false").lower() in ("1", "true", "yes")
UNKNOWN_THRESHOLD = float(os.environ.get("THRESHOLD", "1.0"))

torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("face_recognition")
logger.setLevel(logging.INFO)

_sqs = boto3.client("sqs", region_name=AWS_REGION)

def send_response_message(payload: dict) -> str:
    if not RESPONSE_QUEUE_URL:
        logger.error("RESPONSE_QUEUE_URL is not configured")
        return None
    try:
        resp = _sqs.send_message(QueueUrl=RESPONSE_QUEUE_URL, MessageBody=json.dumps(payload, separators=(",", ":")))
        return resp.get("MessageId")
    except Exception:
        logger.exception("Failed to send response SQS message")
        return None


TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

def _b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")

@torch.no_grad()
def _embed(img: Image.Image) -> torch.Tensor:
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  
    emb = _resnet(x)  
    return emb.squeeze(0).cpu()

def _to_2d512(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.float().cpu()
    if t.ndim == 1 and t.numel() == 512:
        return t.view(1, 512).contiguous()
    if t.ndim == 2 and t.shape[1] == 512:
        return t.contiguous()
    if t.ndim == 3 and t.shape[1] == 1 and t.shape[2] == 512:
        return t.squeeze(1).contiguous()
    try:
        return t.reshape(-1, 512).contiguous()
    except Exception:
        raise ValueError(f"Facebank embeddings have unexpected shape: {tuple(t.shape)}")

def _load_facebank_pt(path: str) -> Tuple[torch.Tensor, list]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FACEBANK_PATH not found: {path}")

    saved = torch.load(path, map_location="cpu")
    if not isinstance(saved, (list, tuple)) or len(saved) < 2:
        raise ValueError(f"Unexpected facebank container in {path}")

    embedding_blob, name_list = saved[0], saved[1]
    labels = [str(x) for x in list(name_list)]

    if isinstance(embedding_blob, list):
        pieces = []
        for e in embedding_blob:
            te = torch.as_tensor(e).float()
            if te.ndim == 1 and te.numel() == 512:
                te = te.view(1, 512)
            elif te.ndim == 2 and te.shape[-1] == 512:
                pass
            elif te.ndim == 3 and te.shape[-2] == 1 and te.shape[-1] == 512:
                te = te.squeeze(-2)
            else:
                try:
                    te = te.reshape(-1, 512)
                except Exception:
                    raise ValueError(f"Facebank element has unexpected shape: {tuple(te.shape)}")
            pieces.append(te)
        embs = torch.vstack(pieces) if pieces else torch.empty((0, 512), dtype=torch.float32)
    else:
        embs = _to_2d512(torch.as_tensor(embedding_blob).float())

    if embs.ndim != 2 or embs.shape[1] != 512:
        raise ValueError(f"Facebank embeddings have unexpected shape: {tuple(embs.shape)}")

    if len(labels) != embs.shape[0]:
        if len(labels) > embs.shape[0]:
            labels = labels[:embs.shape[0]]
        else:
            labels = labels + [f"person_{i}" for i in range(len(labels), embs.shape[0])]

    return embs.contiguous(), labels

try:
    logger.info("Initializing InceptionResnetV1 model (vggface2)")
    _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
except Exception as e:
    logger.exception("Failed to initialize InceptionResnetV1")
    raise

try:
    FACEBANK_EMBS, FACEBANK_LABELS = _load_facebank_pt(FACEBANK_PATH)
    logger.info("Facebank loaded: embeddings=%s labels=%d", tuple(FACEBANK_EMBS.shape), len(FACEBANK_LABELS))
except Exception as e:
    logger.exception("Failed to load facebank")
    raise

FACEBANK_EMBS = FACEBANK_EMBS.float().cpu()

@torch.no_grad()
def _nearest(emb: torch.Tensor) -> Tuple[str, float]:
    if emb.ndim == 1:
        emb = emb.view(1, -1)
    emb = emb.float().cpu()
    dists = torch.cdist(emb.view(1, -1), FACEBANK_EMBS, p=2.0).squeeze(0)
    min_val, idx = torch.min(dists, dim=0)
    label = FACEBANK_LABELS[int(idx)]
    if USE_UNKNOWN_THRESHOLD and float(min_val.item()) > UNKNOWN_THRESHOLD:
        label = "unknown"
    return label, float(min_val.item())

def lambda_handler(event, context):
    processed = 0
    results = []

    records = event.get("Records", []) if isinstance(event, dict) else []
    if not records:
        logger.info("No records found in event")
        return {"statusCode": 200, "body": json.dumps({"processed": 0, "results": []})}

    for rec in records:
        msg_body = rec.get("body")
        try:
            msg = json.loads(msg_body) if isinstance(msg_body, str) else (msg_body or {})
            request_id = msg.get("request_id", "unknown")
            face_b64 = msg.get("face") or msg.get("face_image") or msg.get("content")
            if not face_b64:
                logger.error("No 'face' payload in message for request_id=%s", request_id)
                send_response_message({"request_id": request_id, "result": {"label": None, "error": "missing-face"}})
                results.append({"request_id": request_id, "label": None, "error": "missing-face"})
                continue

            img = _b64_to_pil(face_b64)
            emb = _embed(img)
            label, dist = _nearest(emb)
            payload = {"request_id": request_id, "result": {"label": label, "distance": dist}}
            mid = send_response_message(payload)
            results.append({"request_id": request_id, "label": label, "distance": dist, "sqs_message_id": mid})
            processed += 1
            logger.info("Processed request_id=%s -> %s (dist=%s)", request_id, label, dist)

        except Exception as e:
            logger.exception("Error processing recognition record")
            rid = "unknown"
            try:
                rid = (json.loads(msg_body) if isinstance(msg_body, str) else (msg_body or {})).get("request_id", "unknown")
            except Exception:
                pass
            try:
                send_response_message({"request_id": rid, "result": {"label": "unknown", "error": str(e)}})
            except Exception:
                logger.exception("Failed to enqueue fallback response")
            results.append({"request_id": rid, "label": "unknown", "error": str(e), "trace": traceback.format_exc()})

    return {"statusCode": 200, "body": json.dumps({"processed": processed, "results": results}, separators=(",", ":"))}
