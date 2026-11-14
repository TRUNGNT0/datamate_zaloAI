# RUNBOOK: đóng gói và nộp image DataMate (làm ở máy build)

Hạn: **Test Solution đóng 23:59 15/11/2026** (1 lượt/ngày, không tính vào lượt chính thức); **nộp chính thức 23:59 16/11/2026 (GMT+7)**, tối đa 2 lượt, chỉ lượt cuối được chấm.
Máy build cần: Linux x86_64, Docker + BuildKit, **>= 150 GB trống** (khuyến nghị 200 GB), internet (tải model 19 GB, pip, base image ~10 GB).
Chạy thử thật cần thêm GPU Blackwell (RTX 5090, driver R570+) hoặc ít nhất một GPU >= 24 GB; không có GPU vẫn build/save được.

## 0. Lấy code từ main (repo private)
```bash
export GH_TOKEN='<token GitHub của bạn>'          # không ghi token vào remote URL
AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
git -c "http.extraHeader=Authorization: Basic $AUTH" clone https://github.com/Oszumeiii/ZAIC_2026.git
cd ZAIC_2026
# lần sau chỉ cần cập nhật:
git -c "http.extraHeader=Authorization: Basic $AUTH" pull --ff-only origin main
git log -1 --oneline            # ghi lại commit dùng để build
```

## 1. Kiểm môi trường (phải "đủ điều kiện")
```bash
export TEAM=ten_doi_viet_thuong            # định danh đội, chữ thường; tên image: zaic2026_datamate_$TEAM:final
bash submission/scripts/00_check_env.sh
```

## 2. Chuẩn bị ngữ cảnh build (tải model đúng revision, chép dữ liệu train, sinh data_manifest.json)
```bash
bash submission/scripts/01_prepare.sh
```

## 3. Build và kiểm image
```bash
bash submission/scripts/02_build.sh          # tùy chọn: INSTALL_FLA=1 bash submission/scripts/02_build.sh
```

## 4. Chạy thử đúng lệnh chấm (cần GPU)
```bash
# 10 câu đầu của public test; in thời gian và kiểm định dạng đầu ra (submission.jsonl + logs)
bash submission/scripts/03_smoke.sh dataset/public_test 10
```
Đọc dòng "Thời gian ... cho 10 câu". Ngoại suy: tổng giờ = nạp model + manifest + số câu x thời gian mỗi câu; private test khoảng 500 câu; **phải < 6 giờ**.

```bash
# chạy ĐẦY ĐỦ hai lần để đo giờ và kiểm tính tái lập (|PA lần 1 - PA lần 2| phải <= 0,03)
bash submission/scripts/04_full_run.sh dataset/public_test run1
bash submission/scripts/04_full_run.sh dataset/public_test run2
python3 submission/scripts/compare_submissions.py run1/submission.jsonl run2/submission.jsonl
# PA trên tập train (có answers.jsonl), vd 60 câu đầu: bước 03 với dataset/train rồi:
docker run --rm --entrypoint python3 -v "$PWD/dataset/train":/data:ro -v "$PWD/submission_out_smoke":/result:ro \
  zaic2026_datamate_$TEAM:final /code/tools/score_submission.py
```
Nếu một lần chạy đầy đủ vượt 6 giờ: giảm `PREPARE_BUDGET_S` hoặc `N_SAMPLES` ở `submission/predict.sh`, build lại.

## 5. Xuất image, nén, MD5
```bash
bash submission/scripts/05_save.sh
```
Ra: `zaic2026_datamate_$TEAM.tar.gz`, `.tar.gz.md5`, `.uncompressed_size_bytes.txt` (phải <= 107374182400). **Không đổi tên** tệp .tar.gz.

## 6. Tải lên và nộp
1. Tải `.tar.gz` lên Google Drive, đặt chia sẻ "Anyone with the link".
2. Ở máy khác tải lại và kiểm: `md5sum -c zaic2026_datamate_$TEAM.tar.gz.md5` phải ra `OK`.
3. Form Solution Submission trên portal DataMate: dán Google URL, MD5 (32 ký tự hex), mô tả giải pháp (tối đa 1000 ký tự), và gửi kèm kích thước .tar chưa nén.
