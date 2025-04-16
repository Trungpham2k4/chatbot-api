from huggingface_hub import HfApi, upload_file

# Đăng nhập bằng token trước đó bằng cách chạy: huggingface-cli login

# Khởi tạo API
api = HfApi()

# Tạo repo (chạy 1 lần, nếu có rồi thì bỏ qua)
try:
    api.create_repo(repo_id="TrungPham132313/phobert2bartpho_full_model", private=False)
    print("✅ Repo created.")
except Exception as e:
    print("ℹ️ Repo may already exist:", e)

# Upload file
upload_file(
    path_or_fileobj="phobert2bartpho_full_model.pt",
    path_in_repo="phobert2bartpho_full_model.pt",
    repo_id="TrungPham132313/phobert2bartpho_full_model",
    repo_type="model"
)

print("✅ Upload completed!")
