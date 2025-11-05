# Face Recognition Service (FRS)

This project implements an end-to-end Face Detection and Recognition microservice using FastAPI, InsightFace, and PyTorch. It is optimized for CPU inference and packaged for Docker deployment.

The system detects human faces in images or CCTV frames, extracts embeddings, and recognizes known identities from a small gallery. It is designed as a lightweight, production-ready service for applications such as surveillance analytics, access control, or casualty search and rescue systems.

---

## Overview

The Face Recognition Service (FRS) provides an API-based interface for:
- Detecting faces in an image.
- Extracting and storing facial embeddings.
- Matching and recognizing faces against a stored gallery of known identities.
- Managing identities and embeddings in a database.

The service runs entirely on CPU and can be easily deployed in a containerized environment.

---

## Features

- **Face Detection** using pre-trained InsightFace models.
- **Face Recognition** with embeddings extracted via ArcFace/AdaFace models.
- **Configurable Similarity Thresholds** for precise matching.
- **RESTful API Endpoints** built using FastAPI, documented with Swagger.
- **Persistent Storage** of embeddings and metadata in SQLite.
- **Dockerized Deployment** for portability and scalability.
- **Extensible Architecture** allowing integration with external applications.

---

## Folder Structure

```
face-recognition-service/
├─ data/
│  ├─ raw/                # raw dataset images
│  ├─ aligned/            # aligned and normalized face crops
│  ├─ gallery/            # gallery images per identity
│  └─ gallery_embeddings/  # stored embeddings database
├─ src/
│  ├─ detector/            # face detection implementation
│  ├─ embedder/            # embedding extraction logic
│  ├─ matcher/             # face matching and similarity search
│  ├─ api/                 # FastAPI application and endpoints
│  ├─ db/                  # database schema and access utilities
│  ├─ utils/               # alignment and preprocessing scripts
│  └─ scripts/             # helper and benchmarking scripts
├─ experiments/
│  ├─ train_logs/          # optional fine-tuning logs
│  └─ metrics/             # evaluation metrics
├─ test_images/             # example images for testing
├─ Dockerfile
├─ requirements.txt
├─ README.md
└─ report.md
```

---

## Setup and Installation

### Prerequisites

* Python 3.10 or higher
* Git
* Docker (optional, for container deployment)

### Step 1: Clone the Repository

```bash
git clone git@github.com:jasimmuhammed203/face-recognition-service.git
cd face-recognition-service
```


Step 2: Create and Activate a Virtual Environment

```
python3 -m venv .venv
source .venv/bin/activate  # for macOS/Linux
```
Step 3: Install Dependencies

```
pip install -r requirements.txt
```
Step 4: Run the FastAPI Server
```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Once started, open http://127.0.0.1:8000/docs. This opens the Swagger UI, where all API endpoints can be tested interactively.


API Endpoints and Optimization**

---

## API Endpoints

### Example: Add an Identity

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/add_identity?name=Alice](http://127.0.0.1:8000/add_identity?name=Alice)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_images/alice1.jpg;type=image/jpeg'


```

### Evaluation, Limitations, and Future Improvements**
---

### Accuracy Metrics

* Precision and recall measured on small test samples.
* Identification accuracy (Top-1 and Top-5) for gallery-based recognition.

### Limitations

* Performance may degrade under extreme lighting or occlusion.
* CPU-only mode is optimized for small-scale inference.
* The gallery must be manually curated for consistency.

---

## Future Improvements

* Integration of Faiss for faster embedding retrieval.
* GPU deployment for large-scale recognition.
* Improved handling of low-light and partial-face scenarios.
* Real-time face tracking in video streams.

###Technologies Used
---

## Technologies Used

* Python 3.10
* FastAPI
* PyTorch
* InsightFace
* ONNX Runtime
* SQLite
* Docker

---

## Author

* **Muhammed Jasim**
* B.Tech in Artificial Intelligence & Data Science
* GitHub: [https://github.com/jasimmuhammed203](https://github.com/jasimmuhammed203)
* Email: `muhammedjasim@example.com`
