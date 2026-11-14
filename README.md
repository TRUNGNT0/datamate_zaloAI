# DataMate & ZAIC 2026 Hybrid Pipeline

Dự án này kết hợp nền tảng Text-to-SQL của **DataMate** với hệ thống xử lý dữ liệu, catalog, SQL solver và rulebook an toàn từ **ZAIC 2026** để xây dựng pipeline suy luận SQL hiệu quả cho bài toán truy vấn theo ngữ nghĩa trên dữ liệu doanh nghiệp Việt Nam.

Mục tiêu của dự án là kết hợp:
- khả năng đọc, chuẩn hóa và catalog hóa dữ liệu đa nguồn,
- logic bảo mật và kiểm soát SQL an toàn,
- gợi ý ngữ nghĩa và few-shot retrieval từ DataMate,
- sinh nhiều SQL và chọn đáp án tối ưu theo vote / abstain.

---

## Tóm tắt nhanh

Pipeline được xây dựng theo mô hình hybrid:

| Thành phần | Nguồn | Vai trò trong dự án |
| --- | --- | --- |
| Ingestion + Catalog + Rulebook | ZAIC 2026 | Đọc dữ liệu, chuẩn hóa schema, phát hiện quan hệ, kiểm tra tính đầy đủ và an toàn SQL |
| Semantic views + debt layer + hints | DataMate | Tạo lớp truy vấn chuẩn hóa, gợi ý theo kiểu câu hỏi, thời gian và grain |
| Few-shot retrieval | DataMate | Tìm ví dụ SQL tương tự, từ shop khác, tối ưu cho câu hỏi mới |
| SQL solver + retry + vote | ZAIC 2026 | Sinh nhiều SQL, chạy thử, sửa lỗi và chọn câu trả lời tối ưu |
| Offline training + LoRA | DataMate | Tạo bộ dữ liệu sạch và fine-tune model mà không làm loạn luồng inference |

---

## 1. Kiến trúc tích hợp

### 1.1. Sự kết hợp theo bảng đặc tả nghiệp vụ

| Thành phần | Nguồn tích hợp | Cách thức hoạt động trong dự án |
| --- | --- | --- |
| Đọc và chuẩn hóa CSV / Excel / SQLite; catalog, quan hệ, kiểu dữ liệu, lỗi parse và độ đầy đủ | ZAIC 2026 | Giữ lại pipeline ingestion/catalog làm nguồn dữ liệu chuẩn (`datamate.ingestion`, `datamate.catalog`). Hệ thống đọc đa định dạng, sửa tiêu đề lệch dòng, chuẩn hóa kiểu dữ liệu, theo dõi `completeness.parse_failed_count`, phát hiện PK/FK và lưu trữ vào DuckDB để tối ưu truy vấn. |
| View chuẩn như `v_sales`, `v_orders` và lớp công nợ | DataMate | Xây dựng semantic view và debt layer (`v_debt_orders`, `v_debt_pay`, macro `debt_at(day)`, `debt_new`, `debt_repaid`). Chỉ sử dụng view đã được cache và kiểm thử, tránh để LLM tự sinh view mới trong từng câu hỏi. |
| Rulebook và quy tắc an toàn SQL | ZAIC 2026 | Tích hợp prompt `SQL_SYSTEM` và các kiểm soát an toàn DuckDB: chỉ cho phép `SELECT` read-only, chặn truy cập mạng, giới hạn RAM/temp disk, kiểm tra grain `DISTINCT` mã đơn, xử lý `issue_counts.possible_duplicate`, và phân loại trạng thái đơn đúng nghiệp vụ. |
| Rewrite câu hỏi, giữ câu gốc, xử lý `as_of_date` và thời gian | ZAIC 2026 | Đưa cả `original_question` và `rewritten_question` vào prompt để không làm lệch ý câu hỏi, đồng thời chuẩn hóa `as_of_date` và phân tích khung thời gian với `describe_period` và `viet_time.py`. |
| Hint theo loại truy vấn, mốc ngày, công nợ, grain và entity match | DataMate | Module `datamate.hints` tạo hint có tính quyết định dựa trên dữ liệu/catalog của từng shop: khoảng ngày, grain, số lượng đơn, lợi nhuận, chuỗi entity, từ khóa nghiệp vụ, v.v. |
| Few-shot retrieval | DataMate | Dùng `datamate.fewshot.Bank` để tìm ví dụ train tương tự theo TF-IDF masked n-grams; chỉ lấy SQL đã được chứng minh đúng và từ các shop khác nhau. |
| Sinh nhiều SQL, chạy thử, sửa lỗi và bỏ phiếu | ZAIC 2026 | Giữ nguyên cơ chế SQL solver: sinh nhiều mẫu SQL, chạy thử trên DuckDB với timeout, bắt lỗi, gửi traceback cho LLM sửa lại, chuẩn hóa kết quả bằng `canon()`, và chọn answer bằng vote/abstain. |
| Offline SFT / LoRA / rejection sampling | DataMate | Duy trì luồng offline riêng trong `training/`: chỉ đưa các mẫu SQL lỗi ít nhất một lần về đúng sau khi kiểm thử xác thực vào tập train, sau đó tạo adapter và few-shot bank. |

---

## 2. Cấu trúc thư mục

```text
datamate_zaic_2026/
├── configs/                    # Cache semantic views & debt layers đã kiểm thử
│   └── public_v9/              # _views_<shop>.json, _debt_<shop>.json
├── assets/                     # Few-shot bank và dữ liệu tham chiếu
│   ├── fewshot_bank.jsonl
│   └── fewshot_bank_v2.jsonl
├── adapters/                   # LoRA adapters (sql_rft_v3, sql_v5)
├── dataset/                    # Dữ liệu thi đấu / train
│   ├── train/
│   └── public_test/
├── datamate/                   # Package mã nguồn chính
│   ├── __init__.py
│   ├── config.py               # Cấu hình chung cho toàn bộ pipeline
│   ├── text.py                 # Chuẩn hóa văn bản tiếng Việt
│   ├── submission.py           # Định dạng và validate submission.jsonl
│   ├── scoring.py              # Đánh giá PA (Penalized Accuracy)
│   ├── views.py                # Canonical views và kiểm định dữ liệu
│   ├── debts.py                # Debt layer và macro công nợ
│   ├── hints.py                # Gợi ý theo thời gian / grain / entity
│   ├── fewshot.py              # Truy hồi few-shot tương tự
│   ├── predict.py              # CLI inference
│   ├── notebook.py             # Quản lý notebook/GPU/env
│   ├── ingestion/              # ZAIC: dữ liệu đa nguồn, profiling và chuẩn hóa
│   ├── catalog/                # ZAIC: catalog và phát hiện quan hệ
│   ├── manifests/              # ZAIC: manifest, policy và metadata
│   ├── rules/                  # ZAIC: rulebook và quy tắc nghiệp vụ
│   ├── storage/                # ZAIC: json store nguyên tử
│   ├── tools/                  # SQL executor, UDFs, viet_time, grounding
│   ├── agent/                  # SQLSolver, SQLContextBuilder, cache, pipeline
│   └── llm/                    # Hỗ trợ model local / vLLM / Gemini API
├── app/                        # Alias layer tương thích ngược với datamate
├── evaluation/                 # Chấm điểm và phân tích lỗi
│   ├── run_eval.py
│   └── scoring.py
├── training/                   # Offline rejection sampling + LoRA
│   ├── build_pool.py
│   ├── build_sft.py
│   ├── build_fewshot_bank.py
│   ├── train_lora.py
│   └── merge_lora.py
├── scripts/                    # Script hỗ trợ chạy và kiểm tra
│   ├── run_public.sh
│   ├── run_holdout.sh
│   ├── build_full_notebook.py
│   └── view_ingestion.py
├── notebooks/                  # Notebook cho EDA, inference, evaluation, training
│   ├── 00_full_pipeline.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_inference.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_training_data.ipynb
│   ├── 05_train_lora.ipynb
│   └── colab_pro.ipynb
├── requirements.txt
├── requirements-gpu.txt
├── pyproject.toml
└── README.md
```

---

## 3. Yêu cầu môi trường

- Python 3.10+
- GPU NVIDIA 16GB - 24GB+ khuyến nghị (RTX 3090, 4090, A100, H100,...)
- Tài nguyên lưu trữ đủ cho dataset, cache và checkpoint adapter

### Cài đặt

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# hoặc .venv\Scripts\activate   # Windows

# Cài đặt dependency cơ bản
pip install -r requirements.txt

# Cài đặt dependency dành cho GPU / model / training
pip install -r requirements-gpu.txt

# Tùy chọn: vLLM để tăng tốc inference trên server GPU
pip install vllm>=0.32.0
```

---

## 4. Chạy suy luận (Inference)

### 4.1. CLI

Chạy trên tập `public_test`:

```bash
python -m datamate.predict \
  --data dataset/public_test \
  --output runs/public_run \
  --model Qwen/Qwen3.5-9B \
  --k 3 \
  --agree-min 2 \
  --views cached \
  --views-from configs/public_v9 \
  --fewshot assets/fewshot_bank.jsonl
```

Chạy với adapter LoRA đã huấn luyện:

```bash
python -m datamate.predict \
  --data dataset/public_test \
  --output runs/public_lora \
  --model Qwen/Qwen3.5-9B \
  --lora adapters/sql_rft_v3 \
  --k 8 \
  --agree-min 4
```

### 4.2. Jupyter / Notebook

Khởi động Jupyter lab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Các notebook quan trọng:
- `01_eda.ipynb`: khám phá schema, shop, rulebook và dữ liệu
- `02_inference.ipynb`: chạy pipeline suy luận, sinh `submission.jsonl`
- `03_evaluation.ipynb`: đánh giá PA và phân tích lỗi
- `04_training_data.ipynb`: build dữ liệu huấn luyện từ câu SQL đúng
- `05_train_lora.ipynb`: fine-tune LoRA trên GPU
- `00_full_pipeline.ipynb`: chạy toàn bộ quy trình end-to-end

---

## 5. Quy trình huấn luyện offline (Rejection Sampling)

Nguyên tắc: không trộn bước huấn luyện vào luồng suy luận online.

### Bước 1: Thu thập lời giải đúng

```bash
python -m datamate.predict --data dataset/train --output runs/train_trace --k 8 --agree-min 1
```

### Bước 2: Lọc dữ liệu đúng và build SFT

```bash
python -m training.build_pool --runs runs/train_trace --answers dataset/train/answers.jsonl --output training/pool.jsonl
python -m training.build_sft --pool training/pool.jsonl --output training/sft_train.jsonl
```

### Bước 3: Huấn luyện LoRA

```bash
python -m training.train_lora --data training/sft_train.jsonl --model Qwen/Qwen3.5-9B --output adapters/my_lora
```

### Bước 4: Cập nhật few-shot bank

```bash
python -m training.build_fewshot_bank --pool training/pool.jsonl --output assets/fewshot_bank.jsonl
```

---

## 6. File nộp bài và kiểm định

Sau mỗi lần chạy inference, hệ thống sẽ sinh `submission.jsonl` và tự động kiểm tra:

- mỗi `id` xuất hiện đúng 1 lần,
- định dạng JSON hợp lệ,
- không chứa `NaN` / `Infinity`,
- kiểu dữ liệu phù hợp với chuẩn `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, hoặc `abstain` / `unanswerable`,
- dung lượng file nằm dưới ngưỡng cho phép.

---

## 7. Workflow đề xuất

1. Chuẩn hóa và catalog hóa dữ liệu bằng pipeline ZAIC.
2. Tạo semantic views / debt layer và cache cho từng shop.
3. Tạo hint + few-shot retrieval để tăng chất lượng prompt.
4. Sinh nhiều SQL bằng solver và chọn bằng vote/abstain.
5. Ghi lại các mẫu đúng để build pool và huấn luyện LoRA offline.
6. Tái sử dụng adapter mới trong lần inference tiếp theo.

---

## 8. Ghi chú

- Dự án này ưu tiên tính ổn định và độ tin cậy của SQL hơn là việc sinh quá nhiều query ngẫu nhiên.
- Việc dùng `abstain` là một lựa chọn có chủ đích để giảm rủi ro sai đáp án và bảo toàn điểm PA.
- Luồng huấn luyện offline được tách biệt rõ ràng khỏi inference online nhằm tránh lệch mục tiêu và vượt quá giới hạn của việc chạy trên dữ liệu thực tế.

Nếu bạn muốn, tôi có thể tiếp tục viết thêm phiên bản README theo phong cách:
- chuyên nghiệp hơn cho GitHub,
- ngắn gọn hơn cho demo,
- hoặc có thêm banner / sơ đồ kiến trúc / bảng lộ trình phát triển.
