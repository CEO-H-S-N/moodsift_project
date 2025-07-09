# MoodSift - AI-Powered Review Sentiment Analyzer

![Project Banner](https://via.placeholder.com/800x200?text=MoodSift+Sentiment+Analysis)

## 📌 Brief Summary
MoodSift detects **nuanced emotions** (sarcasm, frustration, etc.) in social media/product reviews using a fine-tuned RoBERTa model (85% accuracy). It automates data collection from Reddit/Twitter, analyzes sentiment, and visualizes trends via an interactive dashboard.

Key Workflow:
1. **Collect** posts via APIs (500+/day)
2. **Analyze** text with custom ML model
3. **Visualize** results in real-time

---

## 📂 File Structure
```txt
moodsift/
├── app/ # Streamlit frontend
│ ├── main.py # Dashboard entry point
│ ├── components/ # UI modules
│ └── utils.py # Helpers
├── config/ # API/model settings
├── data/ # Raw/processed data
├── pipelines/ # Data processing
│ ├── data_collection.py # Reddit/Twitter API
│ ├── preprocessing.py # Text cleaning
│ └── training.py # Model training
├── services/ # Core logic
│ ├── analysis.py # Sentiment prediction
│ └── storage.py # Data versioning
├── tests/ # Unit tests
└── requirements.txt # Dependencies
```
---

## 🚀 Core Features
- **5 Emotion Detection**  
  Positive, Negative, Neutral, Sarcasm, Frustration
- **Automated Pipeline**  
  From API collection → analysis → storage
- **Live Dashboard**  
  Trends, viral posts, and sentiment distribution

---

## 🛠️ Quick Start
1. Install: `pip install -r requirements.txt`
2. Add API keys to `config/api_keys.py`
3. Run: `streamlit run app/main.py`