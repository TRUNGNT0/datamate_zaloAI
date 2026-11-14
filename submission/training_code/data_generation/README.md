# Sinh dữ liệu huấn luyện

Dữ liệu SFT (nếu có) được sinh từ CHÍNH model gốc trên dữ liệu train do ban tổ chức cung cấp, theo cách chọn lọc từ chối (rejection sampling):
sinh nhiều lời giải cho mỗi câu train, chỉ giữ lời giải cho đáp án khớp `answers.jsonl` của tập train. Mã: `training_code/src/export_sft.py`,
`build_exemplars.py`; lệnh: `training_code/train.sh` (bước 1). Không dùng public test hay private test, không thu đáp án của câu hỏi test.

Nếu thư mục `/code/models/lora` không tồn tại thì KHÔNG có bước huấn luyện nào ảnh hưởng đến suy luận (xem `../README.md`).
