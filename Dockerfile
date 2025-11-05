# === FINAL WORKING DOCKERFILE ===
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV + InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . /app

# Install Python dependencies (explicit versions for stability)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.0 \
        numpy==1.26.4 \
        opencv-python==4.10.0.84 \
        scikit-learn==1.5.2 \
        tqdm==4.66.5 \
        torch==2.3.1 \
        torchvision==0.18.1 \
        insightface==0.7.3 \
        faiss-cpu==1.8.0 \
        onnxruntime==1.17.1 \
        python-multipart==0.0.9 \
        sqlalchemy==2.0.32 \
        pydantic==2.9.1

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]