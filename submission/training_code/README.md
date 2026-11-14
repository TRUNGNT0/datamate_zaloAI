# training_code: huấn luyện, sinh dữ liệu và bằng chứng kiểm chứng

## Trạng thái huấn luyện của bài nộp này
**Mặc định: KHÔNG có bước huấn luyện hay sinh dữ liệu nào ảnh hưởng đến trọng số dùng khi suy luận.** Suy luận dùng nguyên model gốc
`Qwen/Qwen3.5-9B` (revision cố định trong `data_manifest.json`). Nếu thư mục `/code/models/lora/` (có `adapter_config.json`) tồn tại trong image
thì `predict.sh` gắn adapter đó (gộp vào trọng số lúc nạp); khi đó phần "Adapter LoRA" bên dưới được điền đầy đủ và `logs/` chứa nhật ký huấn luyện.

## Nội dung
| Mục | Ghi chú |
| --- | --- |
| `src/` | Mã huấn luyện và sinh dữ liệu (`finetune_qlora.py`, `export_sft.py`, `build_exemplars.py`) |
| `configs/` | Cài đặt suy luận (`inference.json`) và huấn luyện (`finetune.json`) |
| `data/train/` | Dữ liệu train do ban tổ chức cung cấp (không có public/private test) |
| `data_generation/` | Mô tả cách sinh dữ liệu SFT |
| `logs/` | Nhật ký huấn luyện/sinh dữ liệu (rỗng nếu không huấn luyện) |
| `checkpoints/` | Điểm kiểm chứng (rỗng nếu không huấn luyện) |
| `data_manifest.json` | Đường dẫn, kích thước, SHA-256 của dữ liệu và trọng số; URL + revision công khai của model gốc |
| `requirements.txt` | Phụ thuộc huấn luyện đã ghim |
| `train.sh` | Lệnh tái tạo (cần GPU) |

## Tái tạo
Phần cứng tham chiếu: 1 GPU 40 GB (A100) hoặc 2 × RTX 5090. Chạy `bash /code/training_code/train.sh` từ `/code`; thời gian phụ thuộc số câu
(`--per-biz 40` trên 24 business FIT; thu dữ liệu vài giờ, fine-tune vài giờ). Bước không tất định: lấy mẫu nhiệt độ 0,7 khi thu dữ liệu (seed cố định
theo prompt nhưng kết quả phụ thuộc phần cứng); huấn luyện dùng seed 0.

## Adapter LoRA
Chưa có (xem "Trạng thái huấn luyện"). Khi có: ghi ở đây đường dẫn adapter, số mẫu SFT (train/validation), điểm PA trên holdout so với model gốc.

## Model gốc huấn luyện-chỉ (không bắt buộc cho suy luận)
Mọi trọng số cần cho suy luận nằm sẵn trong image. Model gốc: https://huggingface.co/Qwen/Qwen3.5-9B , revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, giấy phép Apache-2.0; SHA-256 từng tệp trong `data_manifest.json`.
