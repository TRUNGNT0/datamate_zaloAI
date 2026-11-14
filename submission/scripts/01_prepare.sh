#!/usr/bin/env bash
# Chuẩn bị ngữ cảnh build: tải model đúng phiên bản, chép dữ liệu huấn luyện + mã huấn luyện, sinh data_manifest.json.
# Chạy từ GỐC repo sau khi đã `git pull`: bash submission/scripts/01_prepare.sh
set -euo pipefail
MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"     # commit cố định trên Hugging Face (ghi trong README và data_manifest.json)
MODEL_DIR="submission/models/qwen3.5-9b"

echo "== 1/4 Tải $MODEL_REPO @ ${MODEL_REVISION:0:8} (khoảng 19 GB) vào $MODEL_DIR"
python3 -m pip install --quiet --user "huggingface_hub>=1.31" 2>&1 | tail -1 || true
python3 - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("$MODEL_REPO", revision="$MODEL_REVISION", local_dir="$MODEL_DIR",
                         allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "LICENSE", "README.md"])
print("đã tải về:", path)
PY
rm -rf "$MODEL_DIR/.cache"
test -f "$MODEL_DIR/config.json" && test -f "$MODEL_DIR/model.safetensors.index.json" || { echo "THIẾU config/index của model"; exit 1; }

echo "== 2/4 Chép dữ liệu huấn luyện (CHỈ train, không chép public_test/private) vào training_code/data"
rm -rf submission/training_code/data && mkdir -p submission/training_code/data
test -d dataset/train || { echo "THIẾU dataset/train"; exit 1; }
cp -a dataset/train submission/training_code/data/train
rm -f submission/training_code/data/train/.DS_Store submission/training_code/data/train/data/.DS_Store

echo "== 3/4 Chép mã huấn luyện/sinh dữ liệu"
mkdir -p submission/training_code/src
cp -a training/*.py submission/training_code/src/

echo "== 4/4 Sinh data_manifest.json (SHA-256 mọi tệp; vài phút vì có 19 GB trọng số)"
python3 submission/scripts/make_data_manifest.py \
  --repo "$MODEL_REPO" --revision "$MODEL_REVISION" --model-dir "$MODEL_DIR" \
  --data-dir submission/training_code/data --out submission/training_code/data_manifest.json
echo "XONG. Bước tiếp: bash submission/scripts/02_build.sh"
