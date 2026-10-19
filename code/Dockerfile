# =============================================================================
# Drishti-PAM — Dockerfile for the static triage dashboard
# -----------------------------------------------------------------------------
# Packages the dashboard so a SOC engineer or demo audience can
# `docker build` + `docker run` the project without installing Python:
#
#     # from this directory (the one that contains this Dockerfile)
#     docker build -t drishti-pam:latest .
#     docker run --rm -p 8000:8000 drishti-pam:latest
#     # open http://localhost:8000 in a browser
#
# Multi-stage to keep the runtime image small (~50 MB) — the builder image
# pulls in faker / matplotlib / reportlab / tqdm to run `smoke.py --backend
# mock`, then the runtime image copies *only* the rendered `docs/` folder
# and serves it with `python -m http.server`.
#
# This image deliberately ships ONLY the mock-backend output. The Hugging
# Face / Tesla T4 / DistilBERT path requires GPU access and is intended for
# Kaggle, not for desktop containers. See `code/notebooks/cloud_run.ipynb`
# for the cloud reproduction path.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder: generate the dashboard data files
# -----------------------------------------------------------------------------
FROM python:3.13-slim AS builder

LABEL stage=builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install only the four mock-pipeline dependencies (faker for synthetic
# names, matplotlib for figures the dashboard does NOT consume, reportlab
# for the optional PDF export, tqdm for progress bars). Skips torch /
# transformers / datasets / accelerate / huggingface_hub which add ~700 MB
# of CUDA wheels we don't need for the mock backend.
RUN pip install --no-cache-dir \
        "faker>=22.0" \
        "matplotlib>=3.8" \
        "reportlab>=4.0" \
        "tqdm>=4.66"

# Copy the source tree. The .dockerignore next to this file excludes the
# generated artefacts (data/, docs/data/, runs/, results/, checkpoints/)
# and the Hugging Face token so the build context stays small and
# reproducible.
COPY . /build

# Five-stage smoke pipeline (mock backend, 500 sessions for a fast container
# build). Generates sessions.json, labels.json, rule_predictions.json,
# llm_predictions.json, comparison.json, manifest.json, split.json into
# /build/data/. Then the dashboard ETL renders /build/docs/data/*.json.
RUN python smoke.py --backend mock --num-sessions 500 --non-interactive
RUN python docs/build_dashboard.py

# -----------------------------------------------------------------------------
# Stage 2 — runtime: serve the static dashboard
# -----------------------------------------------------------------------------
FROM python:3.13-alpine AS runtime

LABEL org.opencontainers.image.title="drishti-pam" \
      org.opencontainers.image.description="Drishti-PAM — hybrid structural-deviation detection for privileged-access monitoring." \
      org.opencontainers.image.source="https://github.com/camseye/drishti-pam" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Pull only the rendered dashboard out of the builder. The /build/code/
# Python source, the smoke output JSON files, and the entire matplotlib /
# faker / reportlab / tqdm install stay behind in the discarded builder
# layer.
COPY --from=builder /build/docs /app/docs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8000/ || exit 1

CMD ["python", "-m", "http.server", "8000", "--directory", "docs"]
