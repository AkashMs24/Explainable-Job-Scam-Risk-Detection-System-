# ==============================
# Dockerfile for JobGuard AI
# Multi-stage build for minimal image size
# ==============================

# ── Stage 1: Builder ────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_IMPROVED.txt .

# Create wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements_IMPROVED.txt

# ── Stage 2: Runtime ───────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (including OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r jobguard && useradd -r -g jobguard jobguard

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements_IMPROVED.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data && \
    chown -R jobguard:jobguard /app

# Switch to non-root user
USER jobguard

# ── Environment Variables ──────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false

# ── Health Check ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# ── Expose Ports ───────────────────────────────────────────────
EXPOSE 8501  # Streamlit
EXPOSE 8000  # FastAPI (optional)

# ── Run Application ────────────────────────────────────────────
# Default: Run Streamlit app
CMD ["streamlit", "run", "app_IMPROVED.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--logger.level=info"]

# ── Alternative Commands (uncomment to use) ────────────────────
# CMD ["uvicorn", "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000"]
