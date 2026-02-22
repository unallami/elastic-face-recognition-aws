import os
import io
import time
import uuid
import json
import subprocess
from typing import Optional

import boto3
from botocore.exceptions import ClientError

TMP_DIR = "/tmp"

s3  = boto3.client("s3",  region_name="us-east-1")
sqs = boto3.client("sqs", region_name="us-east-1")

REQ_URL: Optional[str]  = None
RESP_URL: Optional[str] = None

def log(msg: str) -> None:
    print(msg, flush=True)


def get_queue_urls() -> bool:
    global REQ_URL, RESP_URL
    try:
        REQ_URL  = sqs.get_queue_url(QueueName="1233809163-req-queue")["QueueUrl"]
        RESP_URL = sqs.get_queue_url(QueueName="1233809163-resp-queue")["QueueUrl"]
        log(f"INFO: Request Q URL:  {REQ_URL}")
        log(f"INFO: Response Q URL: {RESP_URL}")
        return True
    except ClientError as e:
        log(f"FATAL: Failed to resolve SQS queue URLs: {e}")
        return False


def extract_root(key: str) -> str:
    base = os.path.basename(key)
    root, _ = os.path.splitext(base)
    return root


def unique_tmp_path(suffix: str = ".jpg") -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    return os.path.join(TMP_DIR, f"img-{uuid.uuid4().hex}{suffix}")


def out_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        log(f"WARN: head_object unexpected error for {bucket}/{key}: {e}")
        return True


def safe_delete(queue_url: str, receipt_handle: str) -> None:
    try:
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except ClientError as e:
        log(f"WARN: Failed to delete message: {e}")


def extend_visibility(receipt_handle: str, new_timeout: int) -> None:
    try:
        sqs.change_message_visibility(
            QueueUrl=REQ_URL,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=new_timeout
        )
        log(f"INFO: Extended visibility to {new_timeout}s for current message.")
    except ClientError as e:
        log(f"WARN: Failed to extend visibility: {e}")

def retry(callable_fn, what: str):
    attempts = 1 + 2
    for i in range(attempts):
        try:
            return True, callable_fn()
        except ClientError as e:
            if i < attempts - 1:
                log(f"WARN: transient error during {what} (attempt {i+1}/{attempts}): {e}")
                time.sleep(0.4)
                continue
            log(f"ERROR: {what} failed after retries: {e}")
            return False, None
        except Exception as e:
            if i < attempts - 1:
                log(f"WARN: unexpected error during {what} (attempt {i+1}/{attempts}): {e}")
                time.sleep(0.4)
                continue
            log(f"ERROR: {what} failed after retries: {e}")
            return False, None


def run_inference(img_path: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["python3", "face_recognition.py", img_path],
            capture_output=True,
            text=True,
            timeout=25,
            check=True,
        )
        return (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        log("ERROR: Inference timed out.")
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Inference failed. stderr={e.stderr}")
    except Exception as e:
        log(f"ERROR: Unexpected error during inference: {e}")
    return None

def process_message(message: dict) -> None:
    receipt = message["ReceiptHandle"]
    raw_body = (message.get("Body") or "").strip()

    if not raw_body:
        log("WARN: Empty message body; deleting.")
        safe_delete(REQ_URL, receipt)
        return

    key = raw_body
    try:
        maybe = json.loads(raw_body)
        if isinstance(maybe, dict) and "key" in maybe:
            key = str(maybe["key"])
    except Exception:
        pass

    root = extract_root(key)

    if out_exists("1233809163-out-bucket", root):
        log(f"INFO: Output already exists for {root}; deleting request only.")
        safe_delete(REQ_URL, receipt)
        return

    img_tmp = unique_tmp_path(".jpg")

    recv_ts = time.time()
    current_vt = 60

    try:
        ok, _ = retry(lambda: s3.download_file("1233809163-in-bucket", key, img_tmp),
                      f"S3 download {key}")
        if not ok:
            return

        elapsed = time.time() - recv_ts
        if elapsed > 0.60 * current_vt:
            current_vt = min(int(current_vt * 2), 300)
            extend_visibility(receipt, current_vt)

        pred = run_inference(img_tmp)
        if pred is None:
            log(f"ERROR: No prediction for {key}; leaving message for retry.")
            return

        elapsed = time.time() - recv_ts
        if elapsed > 0.60 * current_vt:
            current_vt = min(int(current_vt * 2), 300)
            extend_visibility(receipt, current_vt)

        ok, _ = retry(
            lambda: s3.put_object(
                Bucket="1233809163-out-bucket",
                Key=root,
                Body=pred.encode("utf-8"),
                ContentType="text/plain",
            ),
            f"S3 put_object {root}",
        )
        if not ok:
            return

        elapsed = time.time() - recv_ts
        if elapsed > 0.60 * current_vt:
            current_vt = min(int(current_vt * 2), 300)
            extend_visibility(receipt, current_vt)

        ok, _ = retry(
            lambda: sqs.send_message(
                QueueUrl=RESP_URL,
                MessageBody=pred,
                MessageAttributes={
                    "FileName": {"DataType": "String", "StringValue": root}
                },
            ),
            f"SQS send_message for {root}",
        )
        if not ok:
            return

        safe_delete(REQ_URL, receipt)

    except ClientError as e:
        log(f"ERROR: AWS client error while processing {key}: {e}")
    except Exception as e:
        log(f"ERROR: Unexpected error while processing {key}: {e}")
    finally:
        try:
            if os.path.exists(img_tmp):
                os.remove(img_tmp)
        except Exception:
            pass

def main() -> None:
    if not get_queue_urls():
        return

    log("APP TIER: started")
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=REQ_URL,
                MaxNumberOfMessages=1,  
                WaitTimeSeconds=15,           
                VisibilityTimeout=60,    
                MessageAttributeNames=["All"],
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue
            process_message(msgs[0])

        except ClientError as e:
            log(f"ERROR: SQS polling failed: {e}")
            time.sleep(1)
        except Exception as e:
            log(f"ERROR: Unexpected main loop error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
