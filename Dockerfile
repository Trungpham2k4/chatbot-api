# Sử dụng Python 3.12 làm base image
FROM python:3.12-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp requirements và cài đặt dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn
COPY . . 

# Expose cổng, mặc dù Render sẽ tự động sử dụng biến môi trường PORT
EXPOSE 8000

# Chạy ứng dụng FastAPI với Uvicorn, sử dụng cổng từ biến môi trường PORT của Render
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
