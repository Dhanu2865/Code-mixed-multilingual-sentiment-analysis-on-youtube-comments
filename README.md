<!-- ===================== PROJECT TITLE ===================== -->
# 🎥 Code-Mixed Multilingual YouTube Comment Analyzer  

A multilingual **multi-task transformer model** built using **XLM-RoBERTa**, performing:  
- 🎭 **Sentiment Analysis**  
- ☣️ **Toxicity Detection**  
- ⚠️ **Anomaly Detection**  

on **YouTube comments**, including **code-mixed** (English + regional language) and **multilingual** text.

---

<!-- ===================== OVERVIEW ===================== -->
## 🚀 Overview  

This project builds an **end-to-end AI pipeline** that:  
1. **Fetches YouTube comments** via YouTube Data API.  
2. **Preprocesses text** (emoji handling, cleaning, language detection).  
3. **Generates pseudo-labels** for sentiment & toxicity.  
4. **Trains a multi-task transformer (XLM-R)** for three tasks jointly.  
5. **Evaluates** using F1-score, accuracy, and uncertainty-weighted loss.  
6. **Deploys** via a **Streamlit dashboard** for real-time analysis.  

---

<!-- ===================== FEATURES ===================== -->
## 🧩 Features  

✅ Multi-task learning (sentiment, toxicity, anomaly)  
✅ Cross-lingual XLM-RoBERTa backbone  
✅ Weighted loss via **homoscedastic uncertainty**  
✅ Automatic comment collection  
✅ Streamlit UI for testing  
✅ Modular, clean architecture  

---

<!-- ===================== ARCHITECTURE ===================== -->
## 🧠 Architecture  
         ┌────────────────────────────┐
         │  YouTube API Comment Fetch │
         └──────────────┬─────────────┘
                        │
             ┌──────────▼──────────┐
             │ Preprocessing Layer │
             │ (clean + label data)│
             └──────────┬──────────┘
                        │
            ┌───────────▼───────────┐
            │  XLM-R Base Encoder   │
            │ (shared representations)│
            └──────┬────┬────┬──────┘
                   │    │    │
             ┌─────▼────▼────▼─────┐
             │ Task Heads:          │
             │ Sentiment | Toxicity │
             │ Anomaly             │
             └─────────────────────┘


---

<!-- ===================== LOSS EQUATION ===================== -->
## 🧮 Weighted Multi-Task Loss  

To balance task importance, the model learns **uncertainty weights**:
<img width="765" height="163" alt="image" src="https://github.com/user-attachments/assets/78e35f13-1899-483f-8568-087172f505bf" />


🧠 **Intuition:**  
A task with higher uncertainty (noisier data) contributes less to the total loss.

---

<!-- ===================== STRUCTURE ===================== -->
## 📁 Project Structure  
<img width="491" height="435" alt="image" src="https://github.com/user-attachments/assets/d3a0a73e-5525-4f64-a085-6822264a11c2" />
<!-- ===================================================== -->
<!-- =============== ⚙️ SETUP AND INSTALLATION =============== -->
<!-- ===================================================== -->


## ⚙️ Setup & Installation  

Follow these steps to set up and run the **Code-Mixed Multilingual YouTube Comment Analyzer** on your system:  

### 1️⃣ Clone the repository
git clone https://github.com/Dhanu2865/Code-mixed-multilingual-sentiment-analysis-on-youtube-comments.git
cd Code-mixed-multilingual-sentiment-analysis-on-youtube-comments

### 2️⃣ Create and activate a virtual environment
python -m venv venv

### ▶️ Activate the environment
### On Windows:
venv\Scripts\activate

### On Mac/Linux:
source venv/bin/activate

### 3️⃣ Install all dependencies
pip install -r requirements.txt

### 4️⃣ (Optional) Download trained model weights
### If you’ve stored weights externally (Google Drive / Hugging Face)
python download_weights.py

### 5️⃣ Run the Streamlit web app
streamlit run app/app.py


## Model Output
<img width="992" height="210" alt="image" src="https://github.com/user-attachments/assets/96b50140-8ab3-42ad-8a18-1726ebd39a41" />
<img width="1071" height="273" alt="image" src="https://github.com/user-attachments/assets/085a70dd-e3b5-4cae-89d1-28222c37d864" />

<img width="968" height="346" alt="image" src="https://github.com/user-attachments/assets/3304156a-840d-4cf7-a7ef-5290b889e83e" />

---

---

## 🧰 Requirements  

Make sure you have **Python 3.8+** installed and the following dependencies:  

- 🧠 **torch** — Deep learning framework for model training  
- 🤖 **transformers** — Pretrained XLM-RoBERTa and NLP utilities  
- 📊 **pandas** — Data manipulation and analysis  
- 🔢 **numpy** — Numerical computing support  
- 📈 **matplotlib** — Visualizations and charts  
- ⏳ **tqdm** — Progress bars for training and data processing  
- 🎥 **google-api-python-client** — Fetch YouTube comments via API  
- 🌐 **streamlit** — Frontend web app for model inference  
- 🧮 **scikit-learn** — Metrics, evaluation, and preprocessing utilities  
- 😂 **emoji** — Handle emoji text and tokens  
- 🌍 **langdetect** — Detect language for code-mixed data  
- ☁️ **gdown** — Download pretrained weights from Google Drive  

You can install all dependencies at once using:  
pip install -r requirements.txt

## 💡 Future Work  

- 🌐 Expand to more Indian languages (Tamil, Hindi, etc.)  
- 🧠 Add explainability (attention heatmaps)  
- 📝 Include summarization as an additional downstream task  
- ☁️ Host app on **Streamlit Cloud** or **Hugging Face Spaces**  

---

## 📜 License  

Released under the **MIT License**.  
You are free to use, modify, and distribute this project with proper attribution.

---


