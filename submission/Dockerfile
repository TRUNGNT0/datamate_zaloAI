# Image nộp bài DataMate (Zalo AI Challenge 2026). Build từ GỐC repo:
#   docker build --platform linux/amd64 -f submission/Dockerfile -t zaic2026_datamate_<team>:final .
# Base khuyến nghị của ban tổ chức: hỗ trợ RTX 5090 (CUDA 12.8, torch 2.7.1+cu128). Ta dùng engine transformers, không dùng vLLM.
FROM vllm/vllm-openai:v0.10.1.1
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

WORKDIR /code
COPY submission/requirements-container.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Tùy chọn tăng tốc lớp linear-attention của Qwen3.5 (chưa kiểm trên Blackwell nên mặc định TẮT, dùng bản torch thuần):
ARG INSTALL_FLA=0
RUN if [ "$INSTALL_FLA" = "1" ]; then pip install --no-cache-dir flash-linear-attention==0.6.0; fi

# Tách các COPY để trọng số được cache khi chỉ mã thay đổi
COPY submission/models/ /code/models/
COPY submission/training_code/ /code/training_code/
COPY schemas/ /code/schemas/
COPY evaluation/ /code/evaluation/
COPY app/ /code/app/
COPY submission/scripts/score_submission.py /code/tools/score_submission.py
COPY submission/predict.sh submission/README.md /code/
RUN chmod +x /code/predict.sh

CMD ["bash", "/code/predict.sh"]
