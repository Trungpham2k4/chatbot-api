from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer
import torch
from PhoBertToBartPho import PhoBERT2BARTpho
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import requests
from dotenv import load_dotenv
import uvicorn

load_dotenv()

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "phobert2bartpho_full_model.pt")
MODEL_ID = os.getenv("MODEL_ID")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        try:
            import gdown
        except ImportError:
            os.system("pip install gdown")
            import gdown

        file_id = "1HoGKrL-j87MnZ3KvWNDSF65XWq-dkhV-"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already exists locally.")


# def download_model():
#     headers = {
#         "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
#     }

#     if not os.path.exists(MODEL_PATH):
#         print("⬇️ Downloading model from HuggingFace...")
#         response = requests.get(MODEL_URL, headers=headers, stream=True)
#         if response.status_code == 200:
#             with open(MODEL_PATH, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print("✅ Model downloaded successfully.")
#         else:
#             print(f"❌ Failed to download model. Status code: {response.status_code}")
#             print(response.text)
#     else:
#         print("✅ Model already exists locally.")

# Define the startup event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and tokenizers when the app starts
    await startup_event()
    yield
    # Cleanup can be done here if needed
async def startup_event():
    global model, encoder_tokenizer, decoder_tokenizer

    # Download the model if it doesn't exist
    download_model()
    # Load tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    decoder_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("phobert2bartpho_full_model.pt", map_location=device, weights_only= False)
    model.to(device)
    model.eval()

    # Define the generate method
    def generate(self, question, encoder_tokenizer, decoder_tokenizer, max_length=64):
        self.eval()
        inputs = encoder_tokenizer(question, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.encoder_proj(encoder_outputs.last_hidden_state)
            encoder_outputs_fixed = BaseModelOutput(last_hidden_state=memory)

            generated_ids = self.decoder.generate(
                encoder_outputs=encoder_outputs_fixed,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=2.0
            )
        return decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Attach the generate method to the model
    model.generate = generate.__get__(model, model.__class__)
    print("Model and tokenizers loaded successfully!")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Global variables for the model and tokenizers
model = None
encoder_tokenizer = None
decoder_tokenizer = None

# Define the request model
class Query(BaseModel):
    # Field phải match với tên trong request
    question: str

# Define the chat endpoint
@app.post("/chat")
def chat(query: Query):
    print(f"Received question: {query.question}")
    reply = model.generate(query.question, encoder_tokenizer, decoder_tokenizer)
    return {"reply": reply}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Lấy cổng từ biến môi trường hoặc mặc định 8000
    uvicorn.run(app, host="0.0.0.0", port=port)