#!/usr/bin/env bash
# Разработка/тесты: исходники проекта монтируются поверх /app — без пересборки образа.
#
# Примеры:
#   scripts/dev.sh pytest -q
#   scripts/dev.sh python -m img2stl.smoke --img samples/test.png --out out
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-img2stl:latest}"

# CDI-спека GPU: см. комментарий в run.sh
CDI_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cdi"
CDI_OPTS=()
[ -d "$CDI_DIR" ] && CDI_OPTS=(--cdi-spec-dir="$CDI_DIR")

exec podman run --rm \
    --pull=never \
    --security-opt label=disable \
    --userns=keep-id \
    "${CDI_OPTS[@]}" \
    --device nvidia.com/gpu=all \
    -v img2stl-cache:/cache \
    -v "$PWD":/app \
    -w /app \
    "$IMAGE" "$@"
