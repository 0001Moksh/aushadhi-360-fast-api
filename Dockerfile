# Use official Python runtime
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss, torch, numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ARG TORCH_VERSION=2.2.2
# Add timeout and retries to reduce chance of transient network failures
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 --index-url https://download.pytorch.org/whl/cpu --prefer-binary torch==${TORCH_VERSION}
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Optional health check (requests must be installed)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:8000/docs', timeout=5)"

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
