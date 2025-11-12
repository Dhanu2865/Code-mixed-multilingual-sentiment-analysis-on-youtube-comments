import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from model import MultiTaskXLMR

# ===================== SETUP =====================
API_KEY = "Your api key"
youtube = build('youtube', 'v3', developerKey=API_KEY)
MODEL_PATH = "C:/Users/dhanu/YoutubeCommentSharedModel/fyp/model/best_model_multi_uncertainty.pt"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = MultiTaskXLMR(mode="multi_uncertainty")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ===================== FETCH COMMENTS =====================
def fetch_comments(video_id, limit=200):
    comments = []
    next_page_token = None
    fetched = 0
    while fetched < limit:
        req = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100,
            pageToken=next_page_token, textFormat="plainText"
        )
        res = req.execute()
        for item in res["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet["textDisplay"])
            fetched += 1
            if fetched >= limit:
                break
        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# ===================== PREDICTION =====================
def predict_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    sent_logits = outputs['sentiment_logits']
    tox_logits  = outputs['toxicity_logits']
    anom_logits = outputs['anomaly_logits']

    sent_label = ["Negative", "Neutral", "Positive"][sent_logits.argmax(1).item()]
    tox_label  = "Toxic" if tox_logits.argmax(1).item() == 1 else "Non-Toxic"
    anom_label = "Anomalous" if anom_logits.argmax(1).item() == 1 else "Normal"

    return sent_label, tox_label, anom_label

# ===================== STREAMLIT UI =====================
st.title("🎥 YouTube Comment Analyzer")
st.markdown("Predict **Sentiment**, **Toxicity**, and **Anomaly** using your trained XLM-R model.")

video_id = st.text_input("Enter YouTube Video ID (e.g., Ip2TA2ijDmA):")
if st.button("Fetch and Analyze"):
    with st.spinner("Fetching comments..."):
        comments = fetch_comments(video_id)
    st.success(f"Fetched {len(comments)} comments!")

    data = []
    for c in comments:
        sent, tox, anom = predict_comment(c)
        data.append({"Comment": c, "Sentiment": sent, "Toxicity": tox, "Anomaly": anom})

    df = pd.DataFrame(data)
    st.dataframe(df)

    st.markdown("### 📊 Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())

    st.markdown("### ☣️ Toxicity Distribution")
    st.bar_chart(df["Toxicity"].value_counts())

    st.markdown("### ⚠️ Anomaly Distribution")
    st.bar_chart(df["Anomaly"].value_counts())
