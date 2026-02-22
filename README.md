# Elastic Face Recognition on AWS

A multi-tiered, elastic face recognition application built across four stages, progressing from IaaS to PaaS to edge computing using AWS services.

---

## Architecture Overview

The project is split into two main projects, each with two parts:

| Project | Part | Focus |
|---------|------|-------|
| Project 1 | Part I | Web tier on EC2 with S3 + SimpleDB |
| Project 1 | Part II | App tier with custom autoscaling via SQS |
| Project 2 | Part I | Serverless pipeline with AWS Lambda + ECR |
| Project 2 | Part II | Edge computing with AWS IoT Greengrass + MQTT |

---

## Project 1 — IaaS (EC2, S3, SQS)

### Part I — Web Tier

A Flask-based web server deployed on a single EC2 instance that:
- Accepts concurrent HTTP POST requests on port 8000 with an image payload (`inputFile`)
- Uploads the received image to an S3 input bucket
- Queries Amazon SimpleDB (pre-populated with 1000 face classifications) to emulate model inference
- Returns the result in plain text format: `<filename>:<prediction>`

**Key files:** `project1-iaas/part1-web-tier/server.py`

**AWS Services:** EC2 (t2.micro), S3, SimpleDB

---

### Part II — App Tier + Autoscaling

Extended the web tier to forward requests to a real ML-based app tier via SQS queues. Added a custom autoscaling controller.

**Web Tier (`server.py`):**
- Uploads image to S3 input bucket
- Sends filename to SQS request queue (`<id>-req-queue`)
- Long-polls the SQS response queue (`<id>-resp-queue`) for the result
- Returns result to client once received

**App Tier (`backend.py`):**
- Polls the SQS request queue for new jobs
- Downloads image from S3, runs face recognition inference using a PyTorch model (`face_recognition.py`)
- Stores prediction result in S3 output bucket
- Sends result back via SQS response queue
- Handles visibility timeout extension and retry logic for resilience

**Autoscaling Controller (`controller.py`):**
- Runs as a background process on the web tier instance
- Monitors SQS queue depth (visible + in-flight messages) every 250ms
- Scales up to 15 pre-initialized app-tier EC2 instances (stopped state) based on queue depth
- Scales back to 0 instances within 2 seconds of queue becoming empty
- Built entirely without AWS Auto Scaling — pure boto3 logic

**Performance:** Sub-1.2s average latency across 100 concurrent requests

**Key files:** `project1-iaas/part2-app-tier/server.py`, `controller.py`, `backend.py`

**AWS Services:** EC2, S3, SQS, custom AMI

---

## Project 2 — PaaS + Edge Computing (Lambda, ECR, IoT Greengrass)

### Part I — Serverless Lambda Pipeline

Replaced the EC2 app tier with two AWS Lambda functions containerized with Docker and deployed via Amazon ECR.

**Face Detection Lambda (`fd_lambda.py`):**
- Triggered via HTTP POST on a Lambda Function URL
- Accepts Base64-encoded video frames with a `request_id` and `filename`
- Runs MTCNN (Multi-task Cascaded Convolutional Networks) for face detection using `facenet_pytorch`
- Compresses and encodes detected faces to fit within the 1KB SQS message size limit
- Sends detected faces to SQS request queue to trigger face recognition

**Face Recognition Lambda (`fr_lambda.py`):**
- Triggered by SQS events from the request queue
- Loads pre-trained InceptionResnetV1 (VGGFace2) model and a facebank of embeddings
- Computes 512-dimensional face embeddings and finds the nearest match via L2 distance
- Returns the predicted label and sends result to SQS response queue

**Performance:** Sub-3s average latency across 100 concurrent requests

**Key files:** `project2-paas/part1-lambda/face-detection/fd_lambda.py`, `face-recognition/fr_lambda.py`

**AWS Services:** Lambda, ECR, SQS, Docker

---

### Part II — Edge Computing with IoT Greengrass (Bonus)

Moved face detection off the cloud and onto an edge device using AWS IoT Greengrass, reducing cloud usage and improving latency for frames with no faces.

**Architecture:**
- An IoT client device (EC2, Ubuntu) publishes Base64-encoded video frames to an MQTT topic (`clients/<id>-IoTThing`) using the AWS IoT Device SDK v2
- A Greengrass Core device (EC2, Amazon Linux) runs the face detection component and subscribes to the MQTT topic
- The face detection Greengrass component (`fd_component.py`) processes frames locally using MTCNN (imported from local `facenet_pytorch` code)
- If a face is detected: sends the cropped face to SQS request queue → triggers the face recognition Lambda
- If no face is detected (bonus): directly pushes `"No-Face"` to the SQS response queue, skipping Lambda entirely
- The IoT client polls the SQS response queue for results

**Performance:** Sub-2.5s average latency across 100 concurrent requests (with bonus)

**Key files:** `project2-paas/part2-edge/face-detection/fd_component.py`, `face-recognition/fr_lambda.py`

**AWS Services:** IoT Greengrass v2, IoT Core, MQTT, Lambda, SQS, EC2

---

## Tech Stack

- **Languages:** Python
- **ML Models:** MTCNN (face detection), InceptionResnetV1 / FaceNet (face recognition), VGGFace2 weights
- **AWS Services:** EC2, S3, SQS, SimpleDB, Lambda, ECR, IoT Greengrass, IoT Core
- **Libraries:** PyTorch, facenet-pytorch, boto3, Flask, torchvision, Pillow, OpenCV, awsiotsdk

---

## Repository Structure

```
elastic-face-recognition-aws/
│
├── project1-iaas/
│   ├── part1-web-tier/
│   │   └── server.py               # Flask web server with S3 + SimpleDB
│   └── part2-app-tier/
│       ├── server.py               # Flask web server with SQS pipeline
│       ├── controller.py           # Custom autoscaling controller
│       └── backend.py              # App tier worker with PyTorch inference
│
├── project2-paas/
│   ├── part1-lambda/
│   │   ├── face-detection/
│   │   │   └── fd_lambda.py        # Face detection Lambda (MTCNN)
│   │   └── face-recognition/
│   │       └── fr_lambda.py        # Face recognition Lambda (FaceNet)
│   └── part2-edge/
│       ├── face-detection/
│       │   └── fd_component.py     # Greengrass edge component (MTCNN + MQTT)
│       └── face-recognition/
│           └── fr_lambda.py        # Face recognition Lambda (FaceNet)
│
└── README.md
```

---

## Notes

- All AWS resources were deployed in `us-east-1`
- Credentials and `.pem` key files are excluded from this repository
- The autoscaling controller in Project 1 Part II is custom-built and does not use AWS Auto Scaling Groups