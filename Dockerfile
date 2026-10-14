# syntax=docker/dockerfile:1
# =============================================================================
# Single image serving both roles: the FastAPI API and the Celery worker.
# The command chosen in docker-compose decides which role a container plays.
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is used by the container healthcheck; build tooling is needed by a few
# wheels that have no prebuilt distribution for slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

# Dependencies are installed first so code edits do not invalidate this layer.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- Development image: dev dependencies plus hot reload ---------------------
FROM base AS development
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
# The data directory is created in the image so a named volume mounted over it
# inherits this ownership; otherwise Docker creates it root-owned and the
# non-root runtime user cannot write documents.
RUN useradd --create-home --uid 1000 vault     && mkdir -p /srv/app/data/documents     && chown -R vault:vault /srv/app
USER vault
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- Production image --------------------------------------------------------
FROM base AS production
COPY . .
RUN useradd --create-home --uid 1000 vault     && mkdir -p /srv/app/data/documents     && chown -R vault:vault /srv/app
USER vault
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
