#!/usr/bin/env bash
# The s1stream Docker environment on a shared GPU server (developed on pro6000, 10 x RTX PRO 6000): an image and a
# container of your own, named after you, never another user's; one GPU only.
#
#   scripts/docker.sh build              build image s1stream:$USER (again after a uv.lock change)
#   GPU=<index> scripts/docker.sh up     start container s1stream-$USER on that GPU (pick a free one: nvidia-smi)
#   SLOT=<n> ...                         a further container, s1stream-$USER-<n>, e.g. one per GPU for parallel runs
#   scripts/docker.sh exec <cmd...>      run a command in it, e.g. scripts/docker.sh exec scripts/run.sh smoke <config>
#   scripts/docker.sh shell              an interactive shell in it
#   scripts/docker.sh down               stop and remove the container (the image and the data stay)
#
# In the container: S1_MACHINE (default pro6000: runs go to results/pro6000/), the repository at /work, DATA (default
# /data/$USER) at the same path, S1_ASSETS=$DATA/s1-assets,
# the Hugging Face cache in $DATA/hf-cache. Long runs: start them in tmux inside the container (scripts/docker.sh shell).
set -euo pipefail
cd "$(dirname "$0")/.."
user=$(id -un)
IMAGE="s1stream:$user"
NAME="s1stream-$user${SLOT:+-$SLOT}"
DATA="${DATA:-/data/$user}"

case "${1:-}" in
    build)
        docker build -f docker/Dockerfile --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t "$IMAGE" . ;;
    up)
        : "${GPU:?set GPU=<index> (one GPU; nvidia-smi shows which are free)}"
        mkdir -p "$DATA/s1-assets" "$DATA/hf-cache"
        docker run -d --name "$NAME" --gpus "device=$GPU" --shm-size 16g \
            -v "$PWD:/work" -v "$DATA:$DATA" \
            -e S1_ASSETS="$DATA/s1-assets" -e HF_HOME="$DATA/hf-cache" -e UCF_CLIPS="$DATA/datasets/ucf-clips" \
            -e S1_MACHINE="${S1_MACHINE:-pro6000}" \
            "$IMAGE" sleep infinity
        docker exec "$NAME" uv sync --frozen --managed-python ;;
    exec)
        shift
        docker exec "$NAME" "$@" ;;
    shell)
        docker exec -it "$NAME" bash ;;
    down)
        docker rm -f "$NAME" ;;
    *)
        sed -n '2,13p' "$0"; exit 1 ;;
esac
