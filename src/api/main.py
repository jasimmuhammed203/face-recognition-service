from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import pickle

app = FastAPI(title="Face Recognition Service", version="1.0")

# Create necessary folders if not exist
os.makedirs("data/gallery_embeddings", exist_ok=True)

# Initialize insightface detector and embedder
model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
model.prepare(ctx_id=0)

# Simple in-memory database (or load from disk)
if os.path.exists("data/gallery_embeddings/embeddings.pkl"):
    with open("data/gallery_embeddings/embeddings.pkl", "rb") as f:
        gallery = pickle.load(f)
else:
    gallery = {}

# ---------- Utility Functions ----------

def read_image(upload_file: UploadFile):
    """Convert UploadFile to OpenCV image (numpy array) safely."""
    data = upload_file.file.read()
    if not data:
        raise ValueError("Empty file uploaded. Please choose a valid image.")
    file_bytes = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it's a valid .jpg or .png file.")
    return img

def save_gallery():
    with open("data/gallery_embeddings/embeddings.pkl", "wb") as f:
        pickle.dump(gallery, f)

# ---------- API Endpoints ----------

@app.get("/")
def home():
    return {"message": "Face Recognition Service is running!"}


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    img = read_image(file)
    faces = model.get(img)
    boxes = [face.bbox.astype(int).tolist() for face in faces]
    return {"num_faces": len(boxes), "boxes": boxes}


@app.post("/add_identity")
async def add_identity(name: str, file: UploadFile = File(...)):
    img = read_image(file)
    faces = model.get(img)
    if not faces:
        return JSONResponse(status_code=400, content={"error": "No face detected"})

    emb = faces[0].normed_embedding
    gallery[name] = emb.tolist()
    save_gallery()

    return {"status": "success", "identity_added": name}


@app.get("/list_identities")
def list_identities():
    return {"identities": list(gallery.keys())}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if not gallery:
        return JSONResponse(status_code=400, content={"error": "No identities in gallery"})

    img = read_image(file)
    faces = model.get(img)
    results = []

    for face in faces:
        emb = face.normed_embedding.reshape(1, -1)
        best_match = None
        best_score = -1

        for name, g_emb in gallery.items():
            g_emb = np.array(g_emb).reshape(1, -1)
            score = cosine_similarity(emb, g_emb)[0][0]
            if score > best_score:
                best_score = score
                best_match = name

        results.append({
            "bbox": face.bbox.astype(int).tolist(),
            "identity": best_match,
            "confidence": float(best_score)
        })

    return {"results": results}