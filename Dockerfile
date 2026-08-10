# Rice Leaf Disease Detection - Docker Container
# v2.0 - addresses AUDIT_BACKEND.md B-07, B-08, B-09, B-11

FROM python:3.10-slim

# B-09: create an unprivileged user up front. v1 ran everything as root.
# UID 1000 matches what Hugging Face Spaces expects.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Build tooling is needed for the pip install but not at runtime, so it is
# removed in the same layer to keep it out of the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y

# B-07: .dockerignore keeps the 3.8GB context (datasets, .git, notebooks) out.
# Only what the app actually serves is copied.
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Only the five checkpoints app.py loads.
#
# Stage 1 = v2, retrained 2026-08-09 on the rebuilt dataset (hard negatives and
# field photographs) to remove the background shortcut.
# Stage 2 = v2, retrained 2026-08-10 on the same splits with healthy and
# not_rice_leaf filtered out. It did not carry the shortcut - that lived in
# Stage 1 - but it had no held-out test set and had never seen a field
# photograph, so it was rebuilt to fix both.
#
# These filenames are pinned deliberately: a wildcard would silently pick up
# whatever a future training run happened to leave in the directory. When
# promoting new weights, update these two lines or the build ships the old ones.
# v1 weights live in saved_models/v1_archive/, excluded via .dockerignore.
COPY saved_models/stage1_models/efficientnet_b3_20260809_202820.pth  saved_models/stage1_models/
COPY saved_models/stage1_models/densenet121_20260809_202820.pth      saved_models/stage1_models/
COPY saved_models/stage1_models/mobilenetv3_20260809_202820.pth      saved_models/stage1_models/
COPY saved_models/stage2_models/vit_base_20260810_050916.pth         saved_models/stage2_models/
COPY saved_models/stage2_models/convnext_tiny_20260810_050916.pth    saved_models/stage2_models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    OMP_NUM_THREADS=2

# B-08: v1 ran `import requests`, which is not in requirements.txt, so the
# healthcheck raised ModuleNotFoundError every time and the container was
# permanently unhealthy. urllib is stdlib.
#
# Hits /health, which now returns 503 when the models are not loaded (B-05),
# so an unhealthy container is actually reported as unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health', timeout=8).status==200 else 1)"

# B-11: one worker, two threads.
#
# Each worker holds the full ~1.2GB model set, so raising --workers multiplies
# memory rather than throughput on a free-tier CPU Space. Threads let a second
# request queue at the WSGI layer instead of being refused, while PyTorch
# inference itself still serialises on the GIL for Python-level work.
# --preload loads models once before forking.
#
# start-period above is 180s because cold model loading on CPU is slow.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", \
     "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--graceful-timeout", "30", \
     "--preload", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "app:app"]
