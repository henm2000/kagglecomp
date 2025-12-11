import sys
import os
# --- 1. SETUP PATHS (The Fix) ---
# Get the folder where THIS script lives (src/TFIDF)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to find the Project Root (ML workshop)
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
# Add project root to sys.path so we can import from 'src'
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

# Silence the "byte fallback" warning (and other Python warnings)
warnings.filterwarnings("ignore")

# Silence the "Some weights were not initialized" message
transformers.logging.set_verbosity_error()

# --- 1. LOAD DATA ---
train = pd.read_csv("data/train.csv")
train = build_text_columns_bert(train)

# --- 2. LABEL ENCODING FOR CATEGORY ---
unique_labels = sorted(train["Category"].unique()) # total 6
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

train["label"] = train["Category"].map(label2id)

# -- 3. TOKENIZER ---
safe_cache_dir = "C:/hf_cache"
MODEL_NAME = "microsoft/deberta-v3-small"
# The 'force_download=True' tells it to ignore the broken files on your PC
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-small",
    cache_dir=safe_cache_dir,
)



#--- 4. DATASET CLASS ---
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
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=safe_cache_dir 
    )
    return model


# --- 7. TRAINING LOOP SKELETON (no training yet) ---
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    print(f"\n===== FOLD {fold+1}/{NUM_FOLDS} =====")

    # Split dataframe using fold indices
    train_df = train.iloc[train_idx]
    valid_df = train.iloc[valid_idx]

    # Create datasets
    train_ds = MAPDataset(train_df, tokenizer)
    valid_ds = MAPDataset(valid_df, tokenizer)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=False)

    # Fresh model for this fold
    model = create_model()

    print(f"Fold {fold+1}: train batches = {len(train_loader)}, valid batches = {len(valid_loader)}")

    # Move model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Number of epochs
    EPOCHS = 8

    # --- 8. TRAINING EPOCHS + VALIDATION + SAVING BEST MODEL ---
    best_val_loss = float("inf")
    patience = 2           # stop after 2 bad epochs
    bad_epochs = 0

    for epoch in range(EPOCHS):
        print(f"\n----- Epoch {epoch+1}/{EPOCHS} -----")

        # ===== TRAINING =====
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

        # ===== CHECK IF BEST MODEL =====
        if avg_val_loss < best_val_loss:
            print("🔥 New best model! Saving checkpoint...")
            best_val_loss = avg_val_loss
            bad_epochs = 0
            
            # Save model for this fold
            save_path = f"models/deberta_category_fold{fold+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            bad_epochs += 1
            print(f"No improvement ({bad_epochs}/{patience} bad epochs)")

        # ===== EARLY STOPPING =====
        if bad_epochs >= patience:
            print("⛔ Early stopping triggered!")
            break