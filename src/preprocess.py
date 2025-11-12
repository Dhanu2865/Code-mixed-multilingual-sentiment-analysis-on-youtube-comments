import pandas as pd
import re
import emoji
from langdetect import detect
from tqdm import tqdm
tqdm.pandas()

def clean_comment(text):
    """Basic comment cleaning."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emojis but keep meaning placeholder
    text = emoji.replace_emoji(text, replace='[emoji]')
    # Remove extra spaces and control characters
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize repeated punctuation (!!! → !)
    text = re.sub(r'([!?.,])\1+', r'\1', text)
    return text

def detect_language_safe(text):
    """Detect language safely; fallback to 'en'."""
    try:
        return detect(text)
    except:
        return "en"

def preprocess_dataset(path):
    df = pd.read_csv(path)
    print(f"🔍 Loaded {len(df)} samples from {path}")

    # Clean text
    df["comment_clean"] = df["comment"].progress_apply(clean_comment)

    # Detect language (optional but useful for cross-lingual)
    df["language"] = df["comment_clean"].progress_apply(detect_language_safe)

    # Encode labels
    sentiment_map = {"positive": 2, "neutral": 1, "negative": 0}
    toxicity_map = {"non-toxic": 0, "toxic": 1}
    anomaly_map = {"normal": 0, "spam": 1}

    df["sentiment_label"] = df["sentiment"].map(sentiment_map).fillna(1).astype(int)
    df["toxicity_label"] = df["toxicity"].map(toxicity_map).fillna(0).astype(int)
    df["anomaly_label"] = df["anomaly"].map(anomaly_map).fillna(0).astype(int)

    # Save cleaned version
    save_path = path.replace(".csv", "_clean.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaned dataset saved to {save_path}")
    return df