import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime

def display_sentiment_metrics(data):
    """Display key sentiment metrics in columns"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Posts", len(data))
    
    with col2:
        positive = len(data[data['sentiment'] == 'positive'])
        st.metric("Positive", f"{positive} ({positive/len(data)*100:.1f}%)")
    
    with col3:
        negative = len(data[data['sentiment'] == 'negative'])
        st.metric("Negative", f"{negative} ({negative/len(data)*100:.1f}%)")
    
    with col4:
        neutral = len(data[data['sentiment'] == 'neutral'])
        st.metric("Neutral", f"{neutral} ({neutral/len(data)*100:.1f}%)")
    
    with col5:
        other = len(data[~data['sentiment'].isin(['positive', 'negative', 'neutral'])])
        st.metric("Other", f"{other} ({other/len(data)*100:.1f}%)")

def display_sentiment_distribution(data):
    """Show sentiment distribution pie chart"""
    st.subheader("Sentiment Distribution")
    fig = px.pie(
        data, 
        names='sentiment', 
        title='Sentiment Distribution',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_trend(data):
    """Show sentiment trend over time"""
    st.subheader("Sentiment Over Time")
    
    if 'created_at' in data.columns:
        data['date'] = pd.to_datetime(data['created_at']).dt.date
        daily_sentiment = data.groupby(['date', 'sentiment']).size().unstack().fillna(0)
        
        # Resample for consistent time intervals
        if len(daily_sentiment) > 7:
            daily_sentiment = daily_sentiment.resample('D').sum()
        
        fig = px.area(
            daily_sentiment,
            title='Sentiment Trend Over Time',
            labels={'value': 'Post Count', 'date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No timestamp data available for trend analysis")

def display_top_posts(data):
    """Show top viral posts based on engagement"""
    st.subheader("Top Viral Posts")
    
    engagement_col = None
    if 'upvotes' in data.columns:
        engagement_col = 'upvotes'
    elif 'likes' in data.columns:
        engagement_col = 'likes'
    elif 'comments' in data.columns:
        engagement_col = 'comments'
    
    if engagement_col:
        top_posts = data.nlargest(5, engagement_col)[['text', engagement_col, 'sentiment']]
        st.dataframe(
            top_posts,
            column_config={
                "text": "Content",
                engagement_col: "Engagement",
                "sentiment": "Sentiment"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No engagement metrics available in this dataset")

def display_raw_data(data):
    """Show raw data with sentiment analysis"""
    st.subheader("Analyzed Data")
    st.dataframe(
        data[['text', 'sentiment', 'sentiment_score', 'created_at']],
        column_config={
            "text": "Content",
            "sentiment": "Sentiment",
            "sentiment_score": "Confidence",
            "created_at": "Date"
        },
        hide_index=True,
        use_container_width=True
    )

def render_dashboard(data):
    """Main dashboard rendering function"""
    if data is not None and not data.empty:
        display_sentiment_metrics(data)
        st.divider()
        display_sentiment_distribution(data)
        display_sentiment_trend(data)
        display_top_posts(data)
        st.divider()
        display_raw_data(data)
    else:
        st.warning("No data available. Please collect data first.")