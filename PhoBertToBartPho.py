import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, get_scheduler
from datasets import load_dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

# Dataset tùy chỉnh
class ViHealthQADataset(Dataset):
    def __init__(self, data, encoder_tokenizer, decoder_tokenizer, max_length=64):
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.pad_token_id = decoder_tokenizer.pad_token_id

    def __getitem__(self, idx):
        item = self.data[idx]
        question_enc = self.encoder_tokenizer(
            item["question"], max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        answer_enc = self.decoder_tokenizer(
            item["answer"], max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": question_enc["input_ids"].squeeze(0),
            "attention_mask": question_enc["attention_mask"].squeeze(0),
            "decoder_input_ids": answer_enc["input_ids"].squeeze(0)[:-1],
            "labels": answer_enc["input_ids"].squeeze(0)[1:]
        }

    def __len__(self):
        return len(self.data)

# Hàm gộp batch
def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "decoder_input_ids": torch.stack([item["decoder_input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# Model kết hợp PhoBERT + BARTpho, fine-tune toàn bộ
class PhoBERT2BARTpho(L.LightningModule):
    def __init__(self, encoder_name="vinai/phobert-base", decoder_name="vinai/bartpho-syllable", lr=5e-5, weight_decay=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_name)

        self.dropout = nn.Dropout(0.1)
        self.encoder_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.d_model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.decoder.config.pad_token_id)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        memory = self.dropout(self.encoder_proj(encoder_outputs.last_hidden_state))

        outputs = self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=(memory, None, None),
            decoder_input_ids=decoder_input_ids
        )
        return outputs.logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"])
        loss = self.criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"])
        val_loss = self.criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        # Scheduler + warmup
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def generate(self, question, encoder_tokenizer, decoder_tokenizer, max_length=64):
        self.eval()
        inputs = encoder_tokenizer(question, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.encoder_proj(encoder_outputs.last_hidden_state)
            generated_ids = self.decoder.generate(
                input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=(memory, None, None),
                max_length=max_length,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=2.0
            )
        return decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Hàm huấn luyện
def train_model():
    encoder_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    decoder_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    ds = load_dataset("tarudesu/ViHealthQA")
    full_data = [{"question": q, "answer": a} for q, a in zip(ds["train"]["question"], ds["train"]["answer"])]

    # Chia dữ liệu: 80% train, 20% val
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_raw, val_raw = random_split(full_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataset = ViHealthQADataset(train_raw, encoder_tokenizer, decoder_tokenizer)
    val_dataset = ViHealthQADataset(val_raw, encoder_tokenizer, decoder_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = PhoBERT2BARTpho()

    # Early stopping nếu val_loss không cải thiện trong 3 epoch
    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        precision="16-mixed" if torch.cuda.is_available() else "bf16-mixed",
        callbacks=[early_stop],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Lưu model
    torch.save(model, "/content/phobert2bartpho_full_model.pt")
    print("✅ Mô hình đã được huấn luyện và lưu vào '/content/phobert2bartpho_full_model.pt'")

if __name__ == "__main__":
    train_model()
