# ---- GPU variant: uncomment the line below and comment the CPU line ----
# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 mluser

# Install Python dependencies (as root, before switching user)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and set ownership
COPY --chown=mluser:mluser . .

# Install package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER mluser

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import torch; print('ok')" || exit 1

ENTRYPOINT ["python", "train.py"]
