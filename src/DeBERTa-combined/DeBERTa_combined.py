import sys
import os

# --- 1. SETUP PATHS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import KFold
from transformers import AutoModelForSequenceClassification
import warnings
import transformers

# Silence warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# --- 1. LOAD DATA ---
print("Loading data...")
train = pd.read_csv("data/train_73k_normalized.csv")

# Build text column from QuestionText, MC_Answer, and StudentExplanation
# Using normalized columns for better quality
train["text"] = (
    train["QuestionText_Norm"].fillna("").astype(str) + " " +
    train["MC_Answer_Norm"].fillna("").astype(str) + " " +
    train["StudentExplanation_Norm"].fillna("").astype(str)
).str.strip()

print(f"Total samples: {len(train)}")

# --- 2. LABEL ENCODING ---
# Use the target column directly as labels
# Examples: "True_Correct:NA", "True_Neither:NA", "False:Misconception_Name"
unique_labels = sorted(train["target"].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

train["label"] = train["target"].map(label2id)

print(f"Total unique target labels: {len(unique_labels)}")
print(f"Sample labels: {list(unique_labels)[:5]}")

# --- 3. TOKENIZER ---
safe_cache_dir = "C:/hf_cache"
MODEL_NAME = "microsoft/deberta-v3-small"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=safe_cache_dir
)

# --- 4. DATASET CLASS ---
class TargetDataset(torch.utils.data.Dataset):
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

# --- 5. MODEL DEFINITION ---
def create_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir=safe_cache_dir
    )
    return model

# --- 6. K-FOLD SETUP ---
NUM_FOLDS = 5
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# --- 7. TRAINING LOOP ---
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{NUM_FOLDS}")
    print(f"{'='*60}")

    # Split dataframe using fold indices
    train_df = train.iloc[train_idx].reset_index(drop=True)
    valid_df = train.iloc[valid_idx].reset_index(drop=True)

    # Create datasets
    train_ds = TargetDataset(train_df, tokenizer)
    valid_ds = TargetDataset(valid_df, tokenizer)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=False)

    # Create fresh model for this fold
    model = create_model()

    # Move model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training parameters
    EPOCHS = 5
    best_val_loss = float("inf")
    patience = 2
    bad_epochs = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # ===== TRAINING =====
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(outputs.logits, dim=1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += len(batch["labels"])

        avg_train_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples * 100
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {accuracy:.2f}%")

        # ===== VALIDATION =====
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_samples = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                valid_loss += outputs.loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                valid_correct += (preds == batch["labels"]).sum().item()
                valid_samples += len(batch["labels"])

        avg_val_loss = valid_loss / len(valid_loader)
        val_accuracy = valid_correct / valid_samples * 100
        
        print(f"Valid Loss: {avg_val_loss:.4f}")
        print(f"Valid Accuracy: {val_accuracy:.2f}%")

        # ===== SAVE BEST MODEL =====
        if avg_val_loss < best_val_loss:
            print("🔥 New best model! Saving...")
            best_val_loss = avg_val_loss
            bad_epochs = 0

            # Create models directory if it doesn't exist
            os.makedirs("models/deBERTa v3 small", exist_ok=True)
            
            # Save model state and mappings
            save_dict = {
                'model_state_dict': model.state_dict(),
                'label2id': label2id,
                'id2label': id2label,
                'num_labels': len(unique_labels)
            }
            
            save_path = f"models/deBERTa v3 small/deberta_combined_fold{fold+1}.pt"
            torch.save(save_dict, save_path)
            print(f"Saved: {save_path}")

        else:
            bad_epochs += 1
            print(f"No improvement ({bad_epochs}/{patience})")

        # ===== EARLY STOPPING =====
        if bad_epochs >= patience:
            print("⛔ Early stopping triggered!")
            break

print("\n" + "="*60)
print("🎉 ALL FOLDS COMPLETE!")
print("="*60)