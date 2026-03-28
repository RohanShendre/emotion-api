FROM python:3.10.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker predict_api:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120"]
