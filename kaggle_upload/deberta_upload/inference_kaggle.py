import pandas as pd
import numpy as np
import torch
from src.preprocess import build_text_columns_bert, build_text_column
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# ===============================
# 1. Load data and Setup Labels
# ===============================
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")

# Preprocess
test = build_text_column(test)
train = build_text_columns_bert(train)

# Category Labels
unique_labels_cat = sorted(train["Category"].unique()) 
label2id_cat = {label: i for i, label in enumerate(unique_labels_cat)}
id2label_cat = {i: label for label, i in label2id_cat.items()}

# Misconception Labels (Filtering out NA to match training)
train_mis = train[train["Misconception"] != "NA"]
unique_labels_mis = sorted(train_mis["Misconception"].unique())  
label2id_mis = {label: i for i, label in enumerate(unique_labels_mis)}
id2label_mis = {i: label for label, i in label2id_mis.items()}

# ===============================
# 2. Load trained models (Offline)
# ===============================
# Path to your metadata (config/tokenizer)
MODEL_NAME = "/kaggle/input/deberta-offline-files/other/default/1"
# Update WEIGHTS_PATH if your .pt files are in a different Input folder
WEIGHTS_PATH = "/kaggle/working/models/deberta" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
config = AutoConfig.from_pretrained(MODEL_NAME)

def load_ensemble(num_labels, l2id, id2l, prefix):
    models = []
    for fold in range(1, 6):
        config.num_labels = num_labels
        config.id2label = id2l
        config.label2id = l2id
        model = AutoModelForSequenceClassification.from_config(config)
        # Using map_location for hardware safety
        state_dict = torch.load(f"{WEIGHTS_PATH}/{prefix}_fold{fold}.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        models.append(model)
    return models

cat_models = load_ensemble(len(unique_labels_cat), label2id_cat, id2label_cat, "deberta_category")
mis_models = load_ensemble(len(unique_labels_mis), label2id_mis, id2label_mis, "deberta_misconception")

# Indices for top-3 logic
cat_classes = unique_labels_cat
mis_classes = unique_labels_mis
TRUE_M_IDX  = list(cat_classes).index("True_Misconception")
FALSE_M_IDX = list(cat_classes).index("False_Misconception")

# ===============================
# 3 & 4. Batched Inference & Prediction Building
# ===============================
BATCH_SIZE = 8 # Small batch to prevent GPU Out-of-Memory
rows = []

print(f"Starting inference on {len(test)} rows...")

for i in range(0, len(test), BATCH_SIZE):
    batch_df = test.iloc[i : i + BATCH_SIZE]
    
    # Tokenize chunk
    inputs = tokenizer(
        batch_df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # Ensemble predictions for Category
        p_cat_batch = np.mean([torch.softmax(m(**inputs).logits, dim=-1).cpu().numpy() for m in cat_models], axis=0)
        # Ensemble predictions for Misconception
        p_mis_batch = np.mean([torch.softmax(m(**inputs).logits, dim=-1).cpu().numpy() for m in mis_models], axis=0)

    # Build top-3 logic for this specific batch
    for b_idx in range(len(batch_df)):
        scores = {}
        
        # (A) Standard Categories
        for j, cat_label in enumerate(cat_classes):
            if cat_label not in ["True_Misconception", "False_Misconception"]:
                scores[f"{cat_label}:NA"] = p_cat_batch[b_idx, j]

        # (B) Misconception Combinations
        p_trueM = p_cat_batch[b_idx, TRUE_M_IDX]
        p_falseM = p_cat_batch[b_idx, FALSE_M_IDX]
        
        for k, mis_label in enumerate(mis_classes):
            scores[f"True_Misconception:{mis_label}"] = p_trueM * p_mis_batch[b_idx, k]
            scores[f"False_Misconception:{mis_label}"] = p_falseM * p_mis_batch[b_idx, k]

        # Sort and take top 3
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_labels = [lbl for lbl, _ in top3]

        rows.append({
            "row_id": batch_df.iloc[b_idx]["row_id"],
            "Category:Misconception": " ".join(top3_labels)
        })

# ===============================
# 5. Save submission.csv
# ===============================
sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")