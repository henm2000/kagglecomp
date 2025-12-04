import pandas as pd
import re

def clean_text(s):
    """Basic cleaning for TF-IDF."""
    if pd.isna(s): 
        return ""
    s = str(s)

    # remove punctuation that doesn't help TF-IDF
    s = re.sub(r'[,!?;"\'\[\]\{\}]', ' ', s)

    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def build_text_column(df):
    """
    Builds the combined 'text' column used by BOTH TF-IDF models.
    This must be identical in training and inference.
    """

    # Convert NaN explanations to empty string
    q = df["QuestionText"].fillna("").apply(clean_text)
    a = df["MC_Answer"].fillna("").apply(clean_text)
    e = df["StudentExplanation"].fillna("").apply(clean_text)

    df["text"] = q + " " + a + " " + e
    return df