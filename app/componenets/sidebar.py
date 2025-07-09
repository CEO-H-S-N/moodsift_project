import streamlit as st
from datetime import datetime, timedelta

def get_time_range_days(time_range_str):
    """Convert time range string to days"""
    if time_range_str == "24 hours":
        return 1
    elif time_range_str == "1 week":
        return 7
    elif time_range_str == "1 month":
        return 30
    elif time_range_str == "3 months":
        return 90
    else:
        return 1

def data_collection_controls():
    """Render data collection controls in sidebar"""
    st.sidebar.header("Data Collection")
    
    source = st.sidebar.selectbox(
        "Select data source",
        ["Reddit", "Twitter"],
        index=0,
        help="Choose between Reddit posts or Twitter tweets"
    )
    
    query = st.sidebar.text_input(
        "Enter search query/subreddit",
        "technology",
        help="For Reddit: subreddit name. For Twitter: search query"
    )
    
    time_range = st.sidebar.selectbox(
        "Time range",
        ["24 hours", "1 week", "1 month", "3 months"],
        index=1,
        help="Time window for data collection"
    )
    
    limit = st.sidebar.slider(
        "Number of posts",
        10, 1000, 100,
        help="Maximum number of posts to analyze"
    )
    
    return {
        "source": source,
        "query": query,
        "time_range": time_range,
        "limit": limit
    }

def analysis_controls():
    """Render analysis controls in sidebar"""
    st.sidebar.header("Analysis Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Minimum confidence threshold",
        0.5, 1.0, 0.7,
        help="Filter out predictions with confidence below this value"
    )
    
    show_details = st.sidebar.toggle(
        "Show detailed analysis",
        value=True,
        help="Display additional charts and metrics"
    )
    
    return {
        "confidence_threshold": confidence_threshold,
        "show_details": show_details
    }

def render_sidebar():
    """Main sidebar rendering function"""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=MoodSift", width=150)
        st.title("Settings")
        
        collection_params = data_collection_controls()
        analysis_params = analysis_controls()
        
        st.divider()
        
        if st.button("Collect and Analyze", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
            st.session_state['collection_params'] = collection_params
            st.session_state['analysis_params'] = analysis_params
        
        st.divider()
        st.caption("ℹ️ Advanced options")
        debug_mode = st.checkbox("Debug mode", False)
        
        if debug_mode:
            st.session_state['debug'] = True
            st.warning("Debug mode enabled")
        else:
            st.session_state['debug'] = False
    
    return {
        "collection": collection_params,
        "analysis": analysis_params
    }