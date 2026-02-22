import time
from typing import List, Tuple

import boto3
from botocore.exceptions import ClientError


sqs = boto3.client("sqs", region_name="us-east-1")
ec2 = boto3.client("ec2", region_name="us-east-1")

_req_url: str = ""


def get_queue_url() -> None:
    global _req_url
    _req_url = sqs.get_queue_url(QueueName="1233809163-req-queue")["QueueUrl"]
    print(f"INFO: SQS Request Queue URL: {_req_url}")


def read_depth() -> Tuple[int, int, int]:
    r = sqs.get_queue_attributes(
        QueueUrl=_req_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )
    vis = int(r.get("Attributes", {}).get("ApproximateNumberOfMessages", "0"))
    invis = int(r.get("Attributes", {}).get("ApproximateNumberOfMessagesNotVisible", "0"))
    return vis, invis, vis + invis


def confirm_idle() -> bool:
    for _ in range(1):
        vis, invis, tot = read_depth()
        if (vis + invis) != 0:
            return False
        time.sleep(0.10)
    return True


def list_pool() -> Tuple[List[str], List[str], List[str]]:
    running, pending, stopped = [], [], []
    filters = [
        {"Name": f"tag:{"Project"}", "Values": ["AppTier"]},
        {"Name": "instance-state-name", "Values": ["running", "pending", "stopped", "stopping"]},
    ]
    res = ec2.describe_instances(Filters=filters)
    for r in res.get("Reservations", []):
        for i in r.get("Instances", []):
            st = i["State"]["Name"]
            iid = i["InstanceId"]
            if st == "running":
                running.append(iid)
            elif st == "pending":
                pending.append(iid)
            elif st == "stopped":
                stopped.append(iid)

    total = len(running) + len(pending) + len(stopped)
    print(
        f"STATUS: Total Managed App Tier Pool: {total} "
        f"(Running: {len(running)}, Pending: {len(pending)}, Stopped: {len(stopped)})"
    )
    return running, pending, stopped


def scale_once(
    vis: int,
    invis: int,
    running: List[str],
    pending: List[str],
    stopped: List[str],
) -> None:

    running_ct = len(running)
    active_ct = running_ct + len(pending)


    desired = min(15, max(1, vis)) if vis > 0 else 0


    if invis > 0:
        if running_ct < 5 and stopped:
            need = min(5 - running_ct, len(stopped))
            to_start = stopped[:need]
            print(f"SCALING: In-flight tail; ensuring keep-warm {5}. Starting {need}.")
            try:
                ec2.start_instances(InstanceIds=to_start)
                print(f"ACTION: start {need} -> {to_start}")
            except ClientError as e:
                print(f"ERROR: start_instances: {e}")
        else:
            print(f"SCALING: In-flight tail; hold. Running={running_ct}, InFlight={invis}")
        return  


    if vis == 0 and invis == 0:
        if confirm_idle():
            if running_ct > 0:
                print(f"SCALING: Queue empty confirmed. Stopping all {running_ct} running.")
                try:
                    ec2.stop_instances(InstanceIds=running)
                    print(f"ACTION: stop {running_ct} -> {running}")
                except ClientError as e:
                    print(f"ERROR: stop_instances: {e}")
            else:
                print("SCALING: Stable idle. Running=0.")
        else:
            print("SCALING: Idle confirm failed; holding briefly.")
        return


    if desired > active_ct and stopped:
        need = min(desired - active_ct, len(stopped))
        to_start = stopped[:need]
        print(f"SCALING: Starting {need} to reach desired={desired} (active={active_ct}).")
        try:
            ec2.start_instances(InstanceIds=to_start)
            print(f"ACTION: start {need} -> {to_start}")
        except ClientError as e:
            print(f"ERROR: start_instances: {e}")
        return


    if vis > 0 and running_ct > desired:
        to_stop = running_ct - desired
        ids = running[:to_stop]
        print(f"SCALING: Reducing running from {running_ct} to {desired}.")
        try:
            ec2.stop_instances(InstanceIds=ids)
            print(f"ACTION: stop {to_stop} -> {ids}")
        except ClientError as e:
            print(f"ERROR: stop_instances: {e}")
        return

    print(
        f"SCALING: Stable-ish. Running={running_ct}, Pending={len(pending)}, "
        f"Visible={vis}, InFlight={invis}, Desired={desired}"
    )


def main() -> None:
    try:
        get_queue_url()
    except ClientError as e:
        print(f"FATAL: get_queue_url failed: {e}")
        return

    print("INFO: Custom Autoscaling Controller started.")
    while True:
        try:
            vis, invis, _ = read_depth()
            running, pending, stopped = list_pool()
            scale_once(vis, invis, running, pending, stopped)

            time.sleep(0.25 if (vis + invis) > 0 else 0.50)
        except Exception as e:
            print(f"ERROR(main): {e}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
