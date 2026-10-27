# The s1stream environment for a shared GPU server (pro6000), built and run by scripts/docker.sh.
# CUDA 12.9 runtime base: torch's cu130 wheels bring their own CUDA libraries and need a driver >= 580. gcc and the
# Python of .python-version (uv-managed, with Python.h): Triton compiles a C helper at run time. The locked
# dependencies, dev group included, in /opt/venv. The repository is mounted at /work at run time, so a code change
# needs no rebuild; a uv.lock change does. The container runs as the host user (UID / GID), so files it writes to the
# mounted folders stay theirs.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ARG UID=1000
ARG GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gcc git libc6-dev tmux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

# The base image's user "ubuntu" has UID 1000; another UID gets a user of its own.
RUN if ! getent passwd "$UID" >/dev/null; then \
        (getent group "$GID" >/dev/null || groupadd -g "$GID" s1) && useradd -m -u "$UID" -g "$GID" s1; \
    fi \
    && mkdir -p /opt/uv /opt/venv /work && chown -R "$UID:$GID" /opt/uv /opt/venv /work

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_CACHE_DIR=/opt/uv/cache \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
USER $UID:$GID
WORKDIR /work

COPY --chown=$UID:$GID .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --managed-python --no-install-project
