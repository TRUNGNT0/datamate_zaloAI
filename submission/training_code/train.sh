#!/usr/bin/env bash
# Điểm vào huấn luyện (CHỈ chạy khi muốn tái tạo adapter LoRA; bài nộp không cần chạy lại, xem README.md).
# Chạy từ /code với dữ liệu train ở training_code/data/train. Cần GPU. Từng bước:
#   1) thu lời giải đúng từ chính model gốc trên các business thuộc nhóm FIT (không đụng test)
#   2) fine-tune LoRA từ sft_train.jsonl
set -euo pipefail
cd /code
export PYTHONPATH=/code:/code/training_code/src
TRAIN=/code/training_code/data/train
OUT=/code/training_code/checkpoints
mkdir -p "$OUT" /code/training_code/logs
python3 -m evaluation.run_eval --train-dir "$TRAIN" --test-dir "$TRAIN" --out-dir "$OUT" --backend local --query-engine pandas \
  --local-model /code/models/qwen3.5-9b --local-engine hf --subset fit --holdout-n 6 --per-biz 40 \
  --n-samples 4 --agree-min 2 --collect-all --export-sft --sft-all --run-name collect1 | tee /code/training_code/logs/collect.log
python3 -m training.finetune_qlora --data "$OUT/sft_train.jsonl" --base /code/models/qwen3.5-9b --out "$OUT/lora" --max-len 8192 \
  | tee /code/training_code/logs/finetune.log
echo "Adapter ở $OUT/lora; để dùng khi suy luận chép sang /code/models/lora (predict.sh tự gắn nếu có adapter_config.json)."
