# NeoSmart — FastAPI web server image.
#
# Runs the parking dashboard on port 8000. Frames, model weights, logs
# and the barrier SQLite DB are mounted in from the host — see
# docker-compose.yml for the canonical wiring.
#
# Build:   docker build -t neosmart:latest .
# Run:     docker compose up
#
# NB: the PlateScanner git submodule must be initialised on the host
# BEFORE `docker build`, otherwise SmartParking/PlateScanner/ is empty
# and plate recognition silently fails. `git submodule update --init
# --recursive` from the repo root does it.

FROM python:3.11-slim AS base

# OpenCV (opencv-python) links against libGL and glib even in headless
# mode; without these two the first `import cv2` dies. build-essential
# stays out — wheels on PyPI are prebuilt for manylinux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Torch is installed separately so the pin lands on the CPU wheel.
# Ultralytics otherwise pulls the full CUDA wheel (~2GB) which this
# image has no use for. GPU deployments should build their own image
# with CUDA base + matching torch wheel.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu

# Requirements first for a cacheable pip layer — project code churns
# much faster than deps, so keep them separate.
COPY SmartParking/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Package metadata + source. `pip install -e .` wires up the
# `neosmart` package so `from neosmart.config import ...` works inside
# web_app/main.py without any sys.path hacks.
COPY pyproject.toml ./
COPY README.md ./
COPY SmartParking ./SmartParking
COPY config ./config
RUN pip install -e .

# Non-root runtime user. Matching UID 1000 to the usual host developer
# UID makes bind-mounted logs/frames writable without chown dance.
RUN useradd --create-home --uid 1000 neosmart \
    && mkdir -p /app/logs /app/SmartParking/web_app/data \
    && chown -R neosmart:neosmart /app
USER neosmart

EXPOSE 8000

# /api/stats is cheap (no YOLO call) and covers "app + config loaded +
# routes wired". Any deeper check would load the detector eagerly.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/stats > /dev/null || exit 1

# web_app/main.py imports siblings (`from detector import ...`) without
# a package prefix, so uvicorn must be launched from that directory.
WORKDIR /app/SmartParking/web_app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
