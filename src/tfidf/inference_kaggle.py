import pandas as pd
import numpy as np
from joblib import load
from src.preprocess import build_text_column

# ===============================
# 1. Load test data
# ===============================
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
test["Misconception"] = "NA"   # placeholder for consistency

# Build the text column (must match training EXACTLY)
test = build_text_column(test)

# ===============================
# 2. Load trained models
# ===============================
tfidf_cat = load("models/tfidf/tfidf_category.pkl")
clf_cat   = load("models/tfidf/clf_category.pkl")
test["Misconception"] = "NA"   # placeholder for consistency

# Build the text column (must match training EXACTLY)
test = build_text_column(test)

# ===============================
# 2. Load trained models
# ===============================
tfidf_cat = load("models/tfidf/tfidf_category.pkl")
clf_cat   = load("models/tfidf/clf_category.pkl")
test["Misconception"] = "NA"   # placeholder for consistency

# Build the text column (must match training EXACTLY)
test = build_text_column(test)

# ===============================
# 2. Load trained models
# ===============================
tfidf_cat = load("models/tfidf/tfidf_category.pkl")
clf_cat   = load("models/tfidf/clf_category.pkl")
test["Misconception"] = "NA"   # placeholder for consistency

# Build the text column (must match training EXACTLY)
test = build_text_column(test)

# ===============================
# 2. Load trained models
# ===============================
tfidf_cat = load("models/tfidf/tfidf_category.pkl")
clf_cat   = load("models/tfidf/clf_category.pkl")
test["Misconception"] = "NA"   # placeholder for consistency

# Build the text column (must match training EXACTLY)
test = build_text_column(test)

# ===============================
# 2. Load trained models
# ===============================
tfidf_cat = load("models/tfidf/tfidf_category.pkl")
clf_cat   = load("models/tfidf/clf_category.pkl")

tfidf_mis = load("models/tfidf/tfidf_misconception.pkl")
clf_mis   = load("models/tfidf/clf_misconception.pkl")

cat_classes = clf_cat.classes_
mis_classes = clf_mis.classes_


# ===============================
# 3. Transform test data
# ===============================
X_cat_test = tfidf_cat.transform(test["text"])
X_mis_test = tfidf_mis.transform(test["text"])

# Predict probabilities
p_cat = clf_cat.predict_proba(X_cat_test)
p_mis = clf_mis.predict_proba(X_mis_test)

# Useful class index lookups
TRUE_M_IDX  = list(cat_classes).index("True_Misconception")
FALSE_M_IDX = list(cat_classes).index("False_Misconception")
# ===============================
# 4. Build top-3 predictions
# ===============================
rows = []
for i in range(len(test)):
    scores = {}

#     # # (A) category-only labels
    # for j, cat_label in enumerate(cat_classes):

    for j, cat_label in enumerate(cat_classes):
        if cat_label in ["True_Misconception", "False_Misconception"]:
            continue  # NEVER allow misconception categories to map to NA
        scores[f"{cat_label}:NA"] = p_cat[i, j]
        scores[f"{cat_label}:NA"] = p_cat[i, j]

    # (B) misconception combinations
    p_trueM  = p_cat[i, TRUE_M_IDX]
    p_falseM = p_cat[i, FALSE_M_IDX]

    for k, mis_label in enumerate(mis_classes):
        scores[f"True_Misconception:{mis_label}"]  = p_trueM  * p_mis[i, k]
        scores[f"False_Misconception:{mis_label}"] = p_falseM * p_mis[i, k]

    # Pick top 3 predictions
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_labels = [lbl for lbl, _ in top3]

    rows.append({
        "row_id": test.iloc[i]["row_id"],
        "Category:Misconception": " ".join(top3_labels)
    })

# ===============================
# 5. Save submission.csv
# ===============================
sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")