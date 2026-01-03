import sys
import os

# --- 1. SETUP PATHS ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import transformers
import numpy as np

# Silence warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# --- 1. LOAD SPELLING CORRECTIONS ---
print("Loading spelling corrections...")
spelling_df = pd.read_csv("data/spelling_corrections_v1.csv")
spelling_dict = dict(zip(spelling_df['misspelled'], spelling_df['correct']))

def normalize_text(text):
    """Normalize text using spelling corrections."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    words = text.split()
    corrected_words = [spelling_dict.get(word, word) for word in words]
    return " ".join(corrected_words)

# --- 3. LOAD TEST DATA ---
print("Loading test data...")
test = pd.read_csv("data/test.csv")
print(f"Test samples: {len(test)}")

# --- 4. NORMALIZE TEST DATA ---
print("Normalizing test data...")
test["QuestionText_Norm"] = test["QuestionText"].apply(normalize_text)
test["MC_Answer_Norm"] = test["MC_Answer"].apply(normalize_text)
test["StudentExplanation_Norm"] = test["StudentExplanation"].apply(normalize_text)

# Build text column
test["text"] = (
    test["QuestionText_Norm"].fillna("").astype(str) + " " +
    test["MC_Answer_Norm"].fillna("").astype(str) + " " +
    test["StudentExplanation_Norm"].fillna("").astype(str)
).str.strip()

# --- 5. TOKENIZER ---
safe_cache_dir = "C:/hf_cache"
MODEL_NAME = "microsoft/deberta-v3-small"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=safe_cache_dir
)

# --- 6. DATASET CLASS ---
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["text"].tolist()
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
            "attention_mask": enc["attention_mask"].squeeze()
        }

# --- 7. LOAD ALL FOLD MODELS AND PREDICT ---
NUM_FOLDS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create test dataset and loader
test_ds = TestDataset(test, tokenizer)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

# Store predictions from all folds
all_probs = []

for fold in range(1, NUM_FOLDS + 1):
    print(f"\n--- Loading Fold {fold}/{NUM_FOLDS} ---")
    
    # Load checkpoint
    checkpoint_path = f"models/deBERTa v3 small/deberta_combined_fold{fold}.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: {checkpoint_path} not found, skipping...")
        continue
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract mappings
    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']
    num_labels = checkpoint['num_labels']
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=safe_cache_dir
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Predict
    fold_probs = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 100 == 0:
                print(f"  Processing batch {i}/{len(test_loader)}")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            
            # Get probabilities using softmax
            probs = torch.softmax(outputs.logits, dim=1)
            fold_probs.append(probs.cpu().numpy())
    
    # Concatenate predictions for this fold
    fold_probs = np.vstack(fold_probs)
    all_probs.append(fold_probs)
    
    print(f"Fold {fold} predictions shape: {fold_probs.shape}")

# --- 8. ENSEMBLE PREDICTIONS (AVERAGE) ---
print("\n--- Creating Ensemble Predictions ---")

# Average probabilities across all folds
avg_probs = np.mean(all_probs, axis=0)  # (num_samples, num_labels)

# Get top 3 predictions for each sample
rows = []
for i in range(len(test)):
    # Get probabilities for this sample
    sample_probs = avg_probs[i]
    
    # Get top 3 indices
    top3_indices = np.argsort(sample_probs)[::-1][:3]
    
    # Convert to labels
    top3_labels = [id2label[idx] for idx in top3_indices]
    
    rows.append({
        'row_id': test.iloc[i]['row_id'],
        'target': ' '.join(top3_labels)
    })

# --- 9. CREATE SUBMISSION ---
print("\n--- Creating Submission ---")

# Create submission dataframe
submission = pd.DataFrame(rows)

# Save submission
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved to: submission.csv")
print(f"Submission shape: {submission.shape}")
print("\nSample predictions:")
print(submission.head(10))

# --- 10. ADDITIONAL STATISTICS ---
print("\n--- Prediction Statistics ---")
print(f"Top 1 Label Distribution:")
top1_labels = [row['target'].split()[0] for row in rows]
target_counts = pd.Series(top1_labels).value_counts().head(10)
for target, count in target_counts.items():
    print(f"  {target}: {count} ({count / len(top1_labels) * 100:.2f}%)")

print("\n✅ Inference Complete!")
