import sys
import os

# --- 1. SETUP PATHS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.preprocess import build_text_columns_bert
from transformers import AutoTokenizer
import torch
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification
import warnings
import transformers
from tqdm import tqdm

# Silence warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# --- 1. LOAD DATA ---
train = pd.read_csv("data/train.csv")
train = build_text_columns_bert(train)

# --- 2. LABEL ENCODING FOR MISCONCEPTION ---
# Misconception has many labels (~32) + NA
train["Misconception"] = train["Misconception"].fillna("NA").astype(str)
train = train[train["Misconception"] != "NA"]

unique_labels = sorted(train["Misconception"].unique())  # ~32 unique labels excluding "NA"

label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

train["label"] = train["Misconception"].map(label2id)

print(f"Total misconception labels: {len(unique_labels)}")

# --- 3. TOKENIZER ---
safe_cache_dir = "C:/hf_cache"
MODEL_NAME = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=safe_cache_dir
)

# --- 4. DATASET CLASS ---
class MAPDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- 5. K-FOLD SETUP ---
NUM_FOLDS = 5
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# --- 6. MODEL DEFINITION ---
def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),    # <-- IMPORTANT: ~32 labels
        id2label=id2label,
        label2id=label2id,
        cache_dir=safe_cache_dir
    )
    return model

# --- 7. TRAINING LOOP ---
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    print(f"\n===== MISCONCEPTION FOLD {fold+1}/{NUM_FOLDS} =====")

    train_df = train.iloc[train_idx]
    valid_df = train.iloc[valid_idx]

    train_ds = MAPDataset(train_df, tokenizer)
    valid_ds = MAPDataset(valid_df, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=False)

    model = create_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    EPOCHS = 8

    best_val_loss = float("inf")
    patience = 2
    bad_epochs = 0

    for epoch in range(EPOCHS):
        print(f"\n----- Epoch {epoch+1}/{EPOCHS} -----")

        # ===== TRAIN =====
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # ===== VALIDATION =====
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                valid_loss += outputs.loss.item()

        avg_val_loss = valid_loss / len(valid_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # ===== BEST MODEL SAVE =====
        if avg_val_loss < best_val_loss:
            print("🔥 New best misconception model! Saving...")
            best_val_loss = avg_val_loss
            bad_epochs = 0

            save_path = f"models/deberta_misconception_fold{fold+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}")

        else:
            bad_epochs += 1
            print(f"No improvement ({bad_epochs}/2)")

        # ===== EARLY STOPPING =====
        if bad_epochs >= patience:
            print("⛔ Early stopping triggered!")
            break

print("\n🔥 ALL MISCONCEPTION FOLDS COMPLETE!")