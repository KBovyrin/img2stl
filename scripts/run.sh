#!/usr/bin/env bash
# Запуск img2stl (контейнер): UI на http://127.0.0.1:7860
#
# Веса моделей живут в named volume img2stl-cache (первый запуск скачивает
# ~5-6 ГБ весов shape-пайплайна + ~170 МБ u2net для rembg, далее офлайн).
# Результаты появляются в ./out на хосте.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-img2stl:latest}"
PORT="${PORT:-7860}"

# CDI-спека GPU: пользовательская (nvidia-ctk cdi generate --output=~/.config/cdi/nvidia.yaml),
# т.к. /etc/cdi/nvidia.yaml на Bazzite может быть устаревшей (сгенерирована под старый драйвер)
CDI_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cdi"
CDI_OPTS=()
[ -d "$CDI_DIR" ] && CDI_OPTS=(--cdi-spec-dir="$CDI_DIR")

mkdir -p out

exec podman run --rm \
    --name img2stl \
    --pull=never \
    --security-opt label=disable \
    --userns=keep-id \
    "${CDI_OPTS[@]}" \
    --device nvidia.com/gpu=all \
    -p "127.0.0.1:${PORT}:7860" \
    -v img2stl-cache:/cache \
    -v "$PWD/out":/app/out \
    "$IMAGE"
