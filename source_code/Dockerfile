FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 讓 gradio 在容器內綁 0.0.0.0、預設 32770
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=32770

WORKDIR /app

# 先安裝需求，加快快取
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 複製整個專案（依你給的結構）
COPY . /app

EXPOSE 32770

# 你的進入點腳本在 agents/app_gradio.py
CMD ["python", "agents/app_gradio.py"]
# CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "32770"]
