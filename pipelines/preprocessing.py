import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from config.settings import PROCESSED_DATA_DIR, MODEL_NAME

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
    def clean_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        text = re.sub(r'\@\w+|\#', '', text)
        # Remove special characters
        text = re.sub(r'\W', ' ', text)
        # Remove single characters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_data(self, df):
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        df = df[df['cleaned_text'].str.len() > 10]  # Remove very short texts
        return df.reset_index(drop=True)
    
    def tokenize_data(self, texts, max_length=128):
        return self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def save_processed_data(self, df, filename):
        df.to_parquet(PROCESSED_DATA_DIR / filename, index=False)