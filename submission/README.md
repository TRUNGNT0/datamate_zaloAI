# DataMate: trả lời câu hỏi tiếng Việt trực tiếp trên dữ liệu bảng (Zalo AI Challenge 2026, Track 2)

Bài nộp Docker: suy luận hoàn toàn ngoại tuyến, một model Qwen3.5-9B, mã nằm ở `/code`. Chạy: `bash /code/predict.sh` (không tham số).
Đọc `/data` (chỉ-đọc), ghi `/result/submission.jsonl` và `/result/logs/`.

## Cách tiếp cận
1. **Đọc dữ liệu.** Mọi tệp `.csv/.xlsx/.sqlite` của từng business được đọc bằng một đường đọc chung (Raw), chuyển kiểu một lần (Typed) rồi
   nạp vào DuckDB cục bộ cùng catalog (tên bảng/cột vật lý, kiểu, hồ sơ cột, quan hệ). Lịch âm-dương, rulebook và README của business được giữ nguyên làm ngữ cảnh.
2. **Manifest do model sinh.** Qwen3.5-9B ánh xạ từng cột nguồn sang ngữ nghĩa nghiệp vụ (đơn vị, vai trò thời gian, cách cộng gộp) theo lô tối đa 12 cột,
   kiểm bằng JSON schema và bằng chứng từ README/rulebook; ô không chắc để "chưa rõ". Hết ngân sách thời gian hoặc lỗi thì business đó dùng manifest nháp tất định (không cần model).
3. **Viết lại câu hỏi.** Model chuẩn hóa tiếng Việt thông tục nhưng giữ nguyên số và mã định danh; lỗi thì giữ câu gốc.
4. **Sinh và chạy SQL.** Với mỗi câu, model sinh JSON `{type, sql, reason}`; SQL chạy trên DuckDB chỉ-đọc (một câu SELECT/WITH, tắt truy cập ngoài, giới hạn bộ nhớ/dòng/thời gian).
   Lỗi thực thi hoặc sai kiểu được trả lại cho model sửa (tối đa 2 lần). Sinh 3 mẫu độc lập; trả lời chỉ khi >= 2 mẫu cho cùng kết quả, ngược lại **abstain**.
   Khi dữ liệu không đủ để trả lời, model trả `unanswerable` kèm lý do.
5. **Ghi đáp án ngay.** `submission.jsonl` được tạo đầy đủ (abstain) lúc bắt đầu rồi cập nhật nguyên tử sau mỗi vài câu; hết 5h30, lỗi hoặc SIGTERM thì phần đã ghi vẫn còn.
6. Không có đáp án gõ cứng và không có logic theo mã câu hỏi/business. Mã chạy được sinh ra là SQL do model viết; log ghi đúng SQL đã chạy.

## Models
| Tên | Nguồn công khai | Revision cố định | Ngày phát hành gốc | Giấy phép | Tham số |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | https://huggingface.co/Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 02/2026 (trước 26/10/2026) | Apache-2.0 | 9.653.104.368 (gồm cả khối thị giác) |

Tổng tham số mọi model dùng khi suy luận: 9,65 tỷ (<= 15 tỷ). Không dùng embedding, reranker hay classifier riêng. Trọng số nạp ở bf16, không nén. Chế độ suy nghĩ tắt
(`enable_thinking=false`), nên trường `thinking` trong log là `null`. Không có adapter LoRA trừ khi tồn tại `/code/models/lora` (xem `training_code/README.md`).

## Dữ liệu
Chỉ dữ liệu train do ban tổ chức cung cấp (`training_code/data/train`: số câu, business và SHA-256 trong `training_code/data_manifest.json`).
Không dùng public/private test để huấn luyện, không thu đáp án câu hỏi test. Không dùng dữ liệu ngoài.

## Cài đặt cố định (trong `predict.sh`)
Model `/code/models/qwen3.5-9b`; nhiệt độ 0,7, top_p 0,8, top_k 20, 3 mẫu, đồng thuận 2; seed 0. Hạt giống mỗi lần sinh suy ra từ nội dung prompt nên chạy lại cùng dữ liệu cho cùng chuỗi lấy mẫu.
Ngân sách: tổng 19.800 giây, trong đó bước manifest tối đa 7.200 giây. Một GPU (`CUDA_VISIBLE_DEVICES=0`).

## Cấu trúc
```
/code/predict.sh            # điểm vào
/code/app/                  # mã suy luận (container_predict.py là điểm vào Python; local_pipeline.py, agent/, ingestion/, manifests/, tools/, llm/)
/code/models/qwen3.5-9b/    # trọng số, tokenizer, cấu hình
/code/schemas/              # JSON schema của manifest/catalog
/code/evaluation/ , /code/tools/score_submission.py   # bộ chấm PA dùng để tự kiểm trên tập train
/code/training_code/        # xem training_code/README.md
```

## Tự kiểm
`docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g --entrypoint bash -v <split>:/data:ro -v <ra>:/result IMAGE /code/predict.sh`
rồi `python3 submission/scripts/check_output.py <split>/questions.jsonl <ra>`; PA trên tập train: `docker run ... --entrypoint python3 IMAGE /code/tools/score_submission.py`.
