import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from services.analysis import SentimentAnalyzer
from pipelines.data_collection import DataCollector
from pipelines.preprocessing import TextPreprocessor
from config.settings import CLASS_NAMES
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from services.storage import DataStorage
from app.utils import timer, validate_sentiment_data

@timer
def process_data(df):
    return validate_sentiment_data(df)

clean_df = process_data(raw_df)

storage = DataStorage()
# Save new data
storage.save_raw_data(df, "twitter")

# Load latest processed reddit data
df = storage.load_latest_data("reddit", processed=True)
def main():
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # When analysis is triggered
    if st.session_state.get('run_analysis', False):
        data = collect_and_analyze_data(params['collection'])
        render_dashboard(data)

if __name__ == "__main__":
    main()

# Initialize services
analyzer = SentimentAnalyzer()
collector = DataCollector()
preprocessor = TextPreprocessor()

# App title
st.title("MoodSift - AI-Powered Review Sentiment Analyzer")

# Sidebar
st.sidebar.header("Data Collection")
source = st.sidebar.selectbox("Select data source", ["Reddit", "Twitter"])
query = st.sidebar.text_input("Enter search query/subreddit", "technology")
time_range = st.sidebar.selectbox("Time range", ["24 hours", "1 week", "1 month"])
limit = st.sidebar.slider("Number of posts", 10, 1000, 100)

if st.sidebar.button("Collect and Analyze"):
    with st.spinner("Collecting data and analyzing sentiment..."):
        # Collect data
        if source == "Reddit":
            data = collector.collect_reddit_posts([query], limit=limit)
        else:
            data = collector.collect_twitter_posts(query, max_results=limit)
        
        # Preprocess data
        data = preprocessor.preprocess_data(data)
        
        # Analyze sentiment
        data['sentiment'] = data['cleaned_text'].apply(
            lambda x: analyzer.get_top_sentiment(x)['label']
        )
        data['sentiment_score'] = data['cleaned_text'].apply(
            lambda x: analyzer.get_top_sentiment(x)['score']
        )
        
        st.session_state['analysis_data'] = data

# Main content
if 'analysis_data' in st.session_state:
    data = st.session_state['analysis_data']
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", len(data))
    col2.metric("Positive Sentiment", 
                f"{len(data[data['sentiment'] == 'positive'])} ({len(data[data['sentiment'] == 'positive'])/len(data)*100:.1f}%)")
    col3.metric("Negative Sentiment", 
                f"{len(data[data['sentiment'] == 'negative'])} ({len(data[data['sentiment'] == 'negative'])/len(data)*100:.1f}%)")
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig = px.pie(data, names='sentiment', title='Sentiment Distribution')
    st.plotly_chart(fig)
    
    # Sentiment over time
    st.subheader("Sentiment Over Time")
    data['date'] = data['created_at'].dt.date
    daily_sentiment = data.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    fig = px.line(daily_sentiment, x=daily_sentiment.index, y=daily_sentiment.columns,
                  title='Sentiment Trend Over Time')
    st.plotly_chart(fig)
    
    # Top viral triggers
    st.subheader("Top Viral Triggers")
    if 'upvotes' in data.columns:
        top_posts = data.nlargest(5, 'upvotes')[['text', 'upvotes', 'sentiment']]
    elif 'likes' in data.columns:
        top_posts = data.nlargest(5, 'likes')[['text', 'likes', 'sentiment']]
    else:
        top_posts = data.sample(5)[['text', 'sentiment']]
    st.dataframe(top_posts)
    
    # Raw data
    st.subheader("Analyzed Data")
    st.dataframe(data[['text', 'sentiment', 'sentiment_score', 'created_at']])
else:
    st.info("Please collect and analyze data using the sidebar controls.")