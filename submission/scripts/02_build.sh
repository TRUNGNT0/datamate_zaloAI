#!/usr/bin/env bash
# Build image từ GỐC repo. Cần: export TEAM=<tên đội viết thường>. Tùy chọn: INSTALL_FLA=1 để cài flash-linear-attention.
set -euo pipefail
: "${TEAM:?Đặt TEAM trước, vd export TEAM=ten_doi}"
IMAGE="zaic2026_datamate_${TEAM}:final"
test -f submission/models/qwen3.5-9b/config.json || { echo "Chưa có model: chạy 01_prepare.sh trước"; exit 1; }
test -f submission/training_code/data_manifest.json || { echo "Chưa có data_manifest.json: chạy 01_prepare.sh trước"; exit 1; }
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg INSTALL_FLA="${INSTALL_FLA:-0}" \
  -f submission/Dockerfile -t "$IMAGE" .
echo "== Kiểm image"
docker image inspect -f '{{.Os}}/{{.Architecture}} {{.Size}} bytes' "$IMAGE"
docker run --rm --network none --entrypoint "" "$IMAGE" find /code -maxdepth 2 | sort | head -40
