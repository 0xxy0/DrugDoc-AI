
# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System dependencies:
# - libgomp1 supports numerical/ML libraries.
# - tesseract-ocr enables OCR for scanned PDFs.
# - tesseract-ocr-eng provides English OCR data.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install the project metadata first to improve build caching.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[ocr]" \
    && python -m pip cache purge

# Create directories required by the application.
RUN mkdir -p \
        /app/data/uploads \
        /app/data/sample_labels \
        /app/data/evaluation \
        /app/data/qdrant \
        /app/reports \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.enableCORS=false"]
