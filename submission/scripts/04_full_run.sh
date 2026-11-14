#!/usr/bin/env bash
# Chạy ĐẦY ĐỦ một split (đúng lệnh chấm, KHÔNG đặt LIMIT) để đo thời gian và so hai lần chạy.
#   bash submission/scripts/04_full_run.sh <thư_mục_split> <thư_mục_ra>
set -euo pipefail
: "${TEAM:?Đặt TEAM trước}"
DATA_SPLIT="$(readlink -f "${1:?split}")"; OUT="$(readlink -f "${2:?thư mục ra}")"; mkdir -p "$OUT"
IMAGE="zaic2026_datamate_${TEAM}:final"
START=$(date +%s)
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash \
  -v "$DATA_SPLIT":/data:ro -v "$OUT":/result "$IMAGE" /code/predict.sh
SECS=$(( $(date +%s) - START ))
N=$(grep -c . "$DATA_SPLIT/questions.jsonl")
echo "== Tổng $SECS giây ($(python3 -c "print(round($SECS/3600,2))") giờ) cho $N câu; giới hạn 6 giờ = 21600 giây"
python3 "$(dirname "$0")/check_output.py" "$DATA_SPLIT/questions.jsonl" "$OUT"
