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

# 1. Load and preprocess full training set
train = pd.read_csv("data/train.csv")
train["Misconception"] = train["Misconception"].fillna("NA").astype(str)

train = build_text_column(train)

# Labels
y_cat = train["Category"].astype(str)

# 2. Build TF-IDF on full training text
tfidf_cat = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words="english",
    lowercase=True,
    strip_accents="unicode",
    sublinear_tf=True,
    min_df=2,
    max_df=0.9
)

X_cat = tfidf_cat.fit_transform(train["text"])

# 3. Train final classifier on ALL data
clf_cat = LogisticRegression(
    max_iter=500,
    n_jobs=-1,
    verbose=1
)
clf_cat.fit(X_cat, y_cat)

print("Training on full dataset complete.")

# 4. Save artifacts
dump(tfidf_cat, "models/TFIDF/tfidf_category.pkl")
dump(clf_cat, "models/TFIDF/clf_category.pkl")

print("\nCategory TF-IDF model training complete.")