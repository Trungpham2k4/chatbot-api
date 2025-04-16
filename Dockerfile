# Sử dụng Python 3.9 làm base image
FROM python:3.12-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . .

# Chạy ứng dụng FastAPI với Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
