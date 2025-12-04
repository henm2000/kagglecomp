###############################################
#need to create pkl files in a kaggle notebook
###############################################
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
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.preprocess import build_text_column


# ============================
# 1. Load and preprocess data
# ============================
train = pd.read_csv("data/train.csv")

# Convert NaN → "NA"
train["Misconception"] = train["Misconception"].fillna("NA").astype(str)

# Build combined text column
train = build_text_column(train)

# Select only rows that actually have a misconception label
train_mis = train[train["Misconception"] != "NA"]

# Labels
y_mis = train_mis["Misconception"].astype(str)


# ============================
# 2. Build TF-IDF Vectorizer
# ============================
tfidf_mis = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words="english",
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    min_df=2,
    max_df=0.9
)

# Fit TF-IDF only on misconception-labeled rows
X_mis = tfidf_mis.fit_transform(train_mis["text"])


# ============================
# 3. Train LogisticRegression
# ============================
clf_mis = LogisticRegression(
    max_iter=500,
    n_jobs=-1,
    verbose=1
)

clf_mis.fit(X_mis, y_mis)

print("Misconception model trained on full misconception-only dataset.")


# ============================
# 4. Save TF-IDF + Model
# ============================
dump(tfidf_mis, "models/TFIDF/tfidf_misconception.pkl")
dump(clf_mis,  "models/TFIDF/clf_misconception.pkl")


print("\nMisconception TF-IDF model training complete.")