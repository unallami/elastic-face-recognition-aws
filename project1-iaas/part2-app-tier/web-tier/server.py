from flask import Flask,request
import boto3,io,os,time
from werkzeug.utils import secure_filename


app = Flask(__name__)


s3 = boto3.client("s3", region_name="us-east-1")
sqs = boto3.client("sqs", region_name="us-east-1")


def extract_root_filename(name: str) -> str:
    base = os.path.basename(name)
    root, _ = os.path.splitext(base)
    return root

req_url = None
resp_url = None

def get_queue_urls():
    global req_url
    global resp_url

    req_url = sqs.get_queue_url(QueueName="1233809163-req-queue")["QueueUrl"]
    resp_url = sqs.get_queue_url(QueueName="1233809163-resp-queue")["QueueUrl"]

def send_recognition_request(filename: str) -> bool:
    message_body = f"{filename}"

    try:
        sqs.send_message(
            QueueUrl=req_url,
            MessageBody=message_body,
            MessageAttributes={
                'FileName': {
                    'StringValue': filename,
                    'DataType': 'String'
                }
            })
        app.logger.info(f"Sent request for {filename}  to SQS.")
        return True

    except Exception as e:
        app.logger.error(f"Failed to send SQS request for {filename}: {e}")
        return False



def receive_recognition_result(expected_root: str, timeout_sec: int = 180) -> str | None:
    global resp_url
    start = time.time()

    def rootify(v: str | None) -> str | None:
        if not v: return None
        b = os.path.basename(v)
        r, _ = os.path.splitext(b)
        return r

    while time.time() - start < timeout_sec:
        try:
            resp = sqs.receive_message(
                QueueUrl=resp_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10,
                MessageAttributeNames=["All"],
            )
            msgs = resp.get("Messages", [])
            if not msgs:
                continue

            m = msgs[0]
            attrs = m.get("MessageAttributes", {}) or {}
            val = None

            for key in ("FileName", "filename", "file_name", "name"):
                if key in attrs and attrs[key].get("DataType") == "String":
                    val = attrs[key]["StringValue"]; break
            if val is None:
                for v in attrs.values():
                    if isinstance(v, dict) and v.get("DataType") == "String":
                        val = v.get("StringValue"); break

            if rootify(val) == expected_root:
                result = m.get("Body", "")
                sqs.delete_message(QueueUrl=resp_url, ReceiptHandle=m["ReceiptHandle"])
                app.logger.info(f"Received & deleted response for root={expected_root}")
                return result

            sqs.change_message_visibility(
                QueueUrl=resp_url,
                ReceiptHandle=m["ReceiptHandle"],
                VisibilityTimeout=0
            )
        except Exception as e:
            app.logger.error(f"Error while polling SQS response queue: {e}")
            time.sleep(0.5)

    return None


@app.post("/")

def upload_and_store():
    if "inputFile" not in request.files:
        return "Input parameter missing: inputFile", 400
    file = request.files["inputFile"]

    if not file.filename:
        return "filename is empty", 400

    safe_name = secure_filename(file.filename)

    try:
        file.stream.seek(0, io.SEEK_SET)
        s3.upload_fileobj(file.stream, "1233809163-in-bucket", safe_name)

    except Exception as e:
        app.logger.error(f"image upload to s3 failed for {safe_name}: {e}")
        return "s3 upload failed", 500

    root = extract_root_filename(safe_name)
    get_queue_urls()

    send_recognition_request(safe_name)
    recognition_result = receive_recognition_result(root)
    print(f"{root}:{recognition_result}")
    return f"{root}:{recognition_result}", 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)



