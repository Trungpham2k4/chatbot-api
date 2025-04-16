FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port for clarity (optional)
EXPOSE 8000

# Chạy bằng shell để biến môi trường PORT được expand
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
