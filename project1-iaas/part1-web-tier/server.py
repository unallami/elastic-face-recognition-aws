from flask import Flask,request
import boto3,io,os
from werkzeug.utils import secure_filename

app = Flask(__name__)

s3 = boto3.client("s3", region_name="us-east-1")
sdb = boto3.client("sdb", region_name="us-east-1")

def extract_root_filename(name: str) -> str:
    base = os.path.basename(name)
    root, _ = os.path.splitext(base)
    return root

def get_result(item_name: str) -> str | None:
    try:
        resp = sdb.get_attributes(
            DomainName="1233809163-simpleDB",
            ItemName=item_name,
            AttributeNames=["recognition"],
            ConsistentRead=True,
        )
        for a in resp.get("Attributes", []):
            if a.get("Name") == "recognition":
                return a.get("Value")
    except Exception as e:
        app.logger.error(f"SDB fetchp failed for {item_name}: {e}")
    return "result not found"


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
    result = get_result(root)

    return f"{root}:{result}", 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
