#!/usr/bin/env bash
# Chạy thử đúng lệnh chấm (2 GPU, không mạng, /data chỉ-đọc, shm 16g) trên N câu đầu rồi kiểm đầu ra.
#   bash submission/scripts/03_smoke.sh <thư_mục_split_có_questions.jsonl> [N=10]
# Ví dụ: bash submission/scripts/03_smoke.sh dataset/public_test 10
set -euo pipefail
: "${TEAM:?Đặt TEAM trước}"
DATA_SPLIT="$(readlink -f "${1:?Tham số 1: thư mục split (có questions.jsonl, data/, rulebook.md, calendar.csv)}")"
N="${2:-10}"
OUT="$(readlink -f "${OUT:-submission_out_smoke}")"; rm -rf "$OUT"; mkdir -p "$OUT"
IMAGE="zaic2026_datamate_${TEAM}:final"
START=$(date +%s)
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -e LIMIT_QUESTIONS="$N" \
  -v "$DATA_SPLIT":/data:ro -v "$OUT":/result \
  "$IMAGE" /code/predict.sh
echo "== Thời gian: $(( $(date +%s) - START )) giây cho $N câu (gồm nạp model + manifest)"
python3 "$(dirname "$0")/check_output.py" "$DATA_SPLIT/questions.jsonl" "$OUT" --limit "$N"
