# DataMate & ZAIC 2026 Hybrid Pipeline

Dự án kết hợp kiến trúc Text-to-SQL cho **Zalo AI Challenge 2026 (Track 2 - DataMate)**, tích hợp khung pipeline chuẩn của **DataMate** cùng hệ thống Ingestion, Catalog, SQL Solver và Rulebook an toàn của **ZAIC_2026** theo bảng đặc tả nghiệp vụ `logout.txt`.

---

## 1. Bảng đối chiếu thành phần kết hợp (Theo logout.txt)

| Thành phần | Nguồn tích hợp | Cách thức hoạt động trong dự án mới |
| :--- | :--- | :--- |
| **1. Đọc và chuẩn hóa CSV/Excel/SQLite; catalog, quan hệ, kiểu dữ liệu, lỗi parse và độ đầy đủ** | **ZAIC_2026** | Giữ làm nguồn dữ liệu chuẩn (`datamate.ingestion`, `datamate.catalog`). Đọc đa định dạng, sửa tiêu đề lệch dòng, chuẩn hóa kiểu dữ liệu, theo dõi `completeness.parse_failed_count`, phát hiện quan hệ PK/FK, lưu trữ vào DuckDB tối ưu. |
| **2. View chuẩn như `v_sales`, `v_orders` và lớp công nợ** | **DataMate** | Xây dựng lớp semantic (`v_sales`, `v_orders`) và debt layer (`v_debt_orders`, `v_debt_pay` + macro `debt_at(day)`, `debt_new`, `debt_repaid`). **Chỉ dùng view đã kiểm thử và lưu/cache** (từ `configs/public_v9` hoặc cache shop), không để LLM tự tạo view mới mỗi lần hỏi. Nếu catalog ZAIC đã có semantic view phù hợp thì dùng nó, tránh tạo trùng. |
| **3. Rulebook và quy tắc an toàn SQL** | **ZAIC_2026** | Tích hợp hệ thống prompt `SQL_SYSTEM` và an toàn DuckDB: chỉ SELECT read-only, chặn truy cập ngoài/mạng, giới hạn RAM (6GB) và temp disk (4GB), kiểm tra grain `DISTINCT` mã đơn, xử lý `issue_counts.possible_duplicate` bằng CTE `DISTINCT *`, phân loại trạng thái đơn (exclude hủy/trả/giao thất bại, include hoàn thành/đang giao). Bổ sung wording nghiệp vụ và luật từ DataMate. |
| **4. Rewrite câu, giữ câu gốc, as_of_date, xử lý thời gian** | **ZAIC_2026** | Đưa cả `original_question` và `rewritten_question` vào prompt để tránh rewrite làm lệch ý. Cung cấp `as_of_date` chuẩn và phân tích khung thời gian (`time_hint`) qua `describe_period` và `viet_time.py`. |
| **5. Hint theo dạng câu hỏi, khoảng ngày, công nợ, grain, giá trị khớp trong dữ liệu** | **DataMate** | Module `datamate.hints` tạo gợi ý tất định cho từng câu hỏi dựa trên dữ liệu/catalog của shop đó: khoảng ngày neo theo `as_of_date`, grain hints (đếm ngày vs đếm đơn, ngày tăng giá, lợi nhuận gộp/ròng, đợt giao), lot stock hint, wording hints ('đh', 'o đờ' = số đơn; 'bữa nay' = today; 'cất hàng' = nhập hàng; 'vay' = 'vậy'), và entity hits khớp chuỗi trực tiếp từ bảng DuckDB của shop. |
| **6. Few-shot theo câu hỏi tương tự** | **DataMate** | Dùng `datamate.fewshot.Bank` để truy hồi top ví dụ train tương tự (TF-IDF masked n-grams). Chỉ lấy ví dụ đã được kiểm chứng SQL đạt gold answer và **tuyệt đối chỉ lấy từ các shop khác** (`business_id != current`), deduplicate theo SQL skeleton. |
| **7. Sinh nhiều SQL, chạy thử, sửa lỗi, bỏ phiếu/abstain** | **ZAIC_2026** | Giữ cơ chế tại `datamate.agent.sql_solver.SQLSolver`: sinh $N$ mẫu SQL qua LLM, chạy thử trên DuckDB có timeout, bắt lỗi và gửi error traceback lại cho LLM tự sửa (tối đa `max_retry` lần), chuẩn hóa kết quả `canon()`, gom phiếu đồng thuận (`agree_min`). Nếu không đạt ngưỡng đồng thuận: hệ thống chọn **abstain** để bảo toàn điểm Penalized Accuracy (PA). |
| **8. SFT/LoRA và rejection sampling** | **DataMate** | Giữ thành pipeline offline riêng biệt trong `training/`: chỉ đưa các mẫu train vào huấn luyện sau khi SQL thực thi thành công và đáp án khớp nhãn chuẩn (`answers.jsonl`). Tạo ra adapter LoRA và few-shot bank; không trộn bước training vào luồng suy luận. |

---

## 2. Cấu trúc thư mục dự án

```text
datamate_zaic_2026/
├── configs/                    # Cache semantic views & debt layers đã kiểm thử
│   └── public_v9/              # _views_<shop>.json, _debt_<shop>.json
├── assets/                     # Kho ví dụ few-shot đã kiểm thử
│   ├── fewshot_bank.jsonl
│   └── fewshot_bank_v2.jsonl
├── adapters/                   # LoRA adapters (sql_rft_v3, sql_v5)
├── dataset/                    # Dữ liệu thi đấu
│   ├── train/                  # data/, answers.jsonl, questions.jsonl, rulebook.md, calendar.csv
│   └── public_test/            # data/, questions.jsonl, rulebook.md, calendar.csv
├── datamate/                   # Package mã nguồn chính
│   ├── __init__.py
│   ├── config.py               # Cấu hình tập trung cho toàn bộ pipeline
│   ├── text.py                 # Chuẩn hóa văn bản tiếng Việt
│   ├── submission.py           # Định dạng, kiểm tra và ghi submission.jsonl
│   ├── scoring.py              # Đánh giá điểm Penalized Accuracy (PA)
│   ├── views.py                # Lớp canonical views (v_sales, v_orders) & data tests
│   ├── debts.py                # Lớp công nợ (v_debt_orders, v_debt_pay) & macro debt_at
│   ├── hints.py                # Gợi ý động theo câu hỏi, grain, kỳ ngày, entity hits
│   ├── fewshot.py              # Truy hồi ví dụ tương tự từ few-shot bank
│   ├── predict.py              # CLI suy luận (python -m datamate.predict)
│   ├── notebook.py             # Quản lý môi trường GPU, venv và đường dẫn notebook
│   ├── ingestion/              # ZAIC: Bộ nạp dữ liệu đa tầng & profiling (CSV/Excel/SQLite)
│   ├── catalog/                # ZAIC: Xây dựng catalog, suy luận quan hệ PK/FK & completeness
│   ├── manifests/              # ZAIC: Quản lý manifest & policy
│   ├── rules/                  # ZAIC: Rulebook parser & business rules
│   ├── storage/                # ZAIC: JsonStore nguyên tử
│   ├── tools/                  # ZAIC: SQLExecutor (UDFs, typed answer), viet_time, grounding
│   ├── agent/                  # ZAIC: SQLSolver, SQLContextBuilder, cache, pipeline
│   └── llm/                    # Hỗ trợ Qwen Local (HF / vLLM) & Gemini API
├── app/                        # Alias layer trỏ về datamate (tương thích ngược 100%)
├── evaluation/                 # Chấm điểm train & phân tích lỗi
│   ├── run_eval.py
│   └── scoring.py
├── training/                   # Pipeline Offline Rejection Sampling & LoRA
│   ├── build_pool.py           # Lọc lời giải đúng từ log chạy train (rejection sampling)
│   ├── build_sft.py            # Dựng tập dữ liệu SFT
│   ├── build_fewshot_bank.py   # Dựng fewshot bank mới
│   ├── train_lora.py           # Huấn luyện LoRA trên Qwen3.5-9B
│   └── merge_lora.py           # Gộp adapter vào model gốc
├── scripts/                    # Scripts tiện ích
│   ├── run_public.sh
│   ├── run_holdout.sh
│   ├── build_full_notebook.py
│   └── view_ingestion.py
├── notebooks/                  # Hệ thống notebook hoàn chỉnh cho Server GPU
│   ├── 00_full_pipeline.ipynb # Toàn bộ quy trình end-to-end
│   ├── 01_eda.ipynb           # Khám phá dữ liệu, schema, shop, rulebook
│   ├── 02_inference.ipynb     # Chạy suy luận sinh submission.jsonl
│   ├── 03_evaluation.ipynb    # Đánh giá PA, nhóm lỗi, tối ưu ngưỡng vote
│   ├── 04_training_data.ipynb # Dựng dữ liệu SFT từ lời giải đúng
│   ├── 05_train_lora.ipynb    # Huấn luyện QLoRA trên GPU
│   └── colab_pro.ipynb        # Chạy pipeline độc lập trên Google Colab Pro
├── requirements.txt            # Thư viện core chạy pipeline
├── requirements-gpu.txt        # Thư viện GPU phục vụ vLLM, transformers, LoRA
└── pyproject.toml
```

---

## 3. Cài đặt trên Server GPU

Yêu cầu: Python 3.10 trở lên, GPU NVIDIA có VRAM từ 16GB - 24GB+ (RTX 3090, 4090, A100, H100, v.v.).

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# hoặc: .venv\Scripts\activate trên Windows

# 2. Cài đặt các thư viện cơ bản
pip install -r requirements.txt

# 3. Cài đặt các thư viện GPU / Model / Training
pip install -r requirements-gpu.txt

# (Tùy chọn cho Linux Server GPU: cài đặt vLLM để tăng tốc suy luận)
pip install vllm>=0.32.0
```

---

## 4. Chạy suy luận (Inference)

### Cách 1: Sử dụng CLI

Chạy suy luận trên tập `public_test` bằng mô hình Qwen local:

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

Nếu muốn chạy bằng adapter LoRA đã huấn luyện:
```bash
python -m datamate.predict \
    --data dataset/public_test \
    --output runs/public_lora \
    --model Qwen/Qwen3.5-9B \
    --lora adapters/sql_rft_v3 \
    --k 8 \
    --agree-min 4
```

### Cách 2: Sử dụng Notebooks trên Server GPU / Jupyter Lab

Khởi động Jupyter Lab trên server:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Mở thư mục `notebooks/`:
- `01_eda.ipynb`: Xem chi tiết schema, cấu trúc file từng shop, rulebook.
- `02_inference.ipynb`: Chạy pipeline suy luận với cấu hình GPU tự động nhận diện, xuất `submission.jsonl`.
- `03_evaluation.ipynb`: Chấm điểm PA (Penalized Accuracy) trên tập train, vẽ biểu đồ phân tích lỗi và điều chỉnh ngưỡng bỏ phiếu (`agree_min`).
- `04_training_data.ipynb`: Rejection sampling — trích xuất các câu SQL chạy đúng đáp án gold để chuẩn bị tập huấn luyện.
- `05_train_lora.ipynb`: Tiến hành QLoRA fine-tuning trên Qwen.
- `00_full_pipeline.ipynb`: Chạy tuần tự từ đầu đến cuối trong 1 notebook duy nhất.

---

## 5. Quy trình huấn luyện LoRA Offline (Rejection Sampling)

Theo nguyên tắc của cuộc thi và thiết kế: **không trộn bước huấn luyện vào luồng suy luận online**.

1. **Thu thập lời giải đúng (Rejection Sampling):**
   Chạy suy luận trên tập train để ghi nhận các mẫu SQL:
   ```bash
   python -m datamate.predict --data dataset/train --output runs/train_trace --k 8 --agree-min 1
   ```
2. **Lọc dữ liệu chính xác (Build Pool & SFT):**
   ```bash
   python -m training.build_pool --runs runs/train_trace --answers dataset/train/answers.jsonl --output training/pool.jsonl
   python -m training.build_sft --pool training/pool.jsonl --output training/sft_train.jsonl
   ```
3. **Huấn luyện QLoRA:**
   ```bash
   python -m training.train_lora --data training/sft_train.jsonl --model Qwen/Qwen3.5-9B --output adapters/my_lora
   ```
4. **Cập nhật Few-shot Bank:**
   ```bash
   python -m training.build_fewshot_bank --pool training/pool.jsonl --output assets/fewshot_bank.jsonl
   ```

---

## 6. Kiểm định file nộp bài (submission.jsonl)

File `submission.jsonl` được kiểm tra tự động sau mỗi lượt chạy:
- Mỗi `id` trong file câu hỏi xuất hiện đúng 1 lần theo thứ tự.
- Đúng định dạng JSON hợp lệ (không chứa `NaN` / `Infinity`).
- Kiểu dữ liệu đáp ứng chuẩn: `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, hoặc `abstain` / `unanswerable`.
- Dung lượng file luôn dưới ngưỡng 5MB.
#   d a t a m a t e _ z a l o A I  
 