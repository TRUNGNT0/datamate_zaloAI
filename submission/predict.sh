#!/usr/bin/env bash
# Điểm vào theo Docker guideline: `bash /code/predict.sh` (không tham số). /data chỉ-đọc, ghi /result/submission.jsonl và /result/logs/.
# Mọi giá trị dưới đây là mặc định CỐ ĐỊNH của bài nộp; biến môi trường chỉ để chạy thử (ban tổ chức không đặt chúng).
set -uo pipefail

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"          # engine transformers dùng một GPU

export DATA_DIR="${DATA_DIR:-/data}" RESULT_DIR="${RESULT_DIR:-/result}" WORK_DIR="${WORK_DIR:-/tmp/zaic_work}"
export MODEL_DIR="${MODEL_DIR:-/code/models/qwen3.5-9b}" LORA_DIR="${LORA_DIR:-/code/models/lora}"
export TOTAL_BUDGET_S="${TOTAL_BUDGET_S:-19800}"      # 5h30 trong giới hạn 6h (đã gồm nạp model)
export PREPARE_BUDGET_S="${PREPARE_BUDGET_S:-7200}"   # tối đa 2h cho bước Qwen sinh manifest, quá hạn dùng manifest nháp tất định
export MARGIN_S="${MARGIN_S:-300}"
export N_SAMPLES="${N_SAMPLES:-3}" AGREE_MIN="${AGREE_MIN:-2}" TEMPERATURE="${TEMPERATURE:-0.7}" SEED="${SEED:-0}"

mkdir -p "$RESULT_DIR/logs" "$WORK_DIR"
cd /code
python3 -m app.container_predict 2>&1 | tee -a "$RESULT_DIR/logs/run.log"
exit "${PIPESTATUS[0]}"
