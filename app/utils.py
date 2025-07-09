import time
import pandas as pd
from functools import wraps
from typing import Callable, Any
from config.settings import CLASS_NAMES
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.runtime.state import SafeSessionState

def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def validate_sentiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean sentiment analysis results.
    Ensures sentiment labels match expected classes and scores are within [0,1].
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_columns = {'text', 'sentiment', 'sentiment_score'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Clean sentiment labels
    df['sentiment'] = df['sentiment'].apply(
        lambda x: x.lower() if isinstance(x, str) else 'neutral'
    )
    df['sentiment'] = df['sentiment'].apply(
        lambda x: x if x in CLASS_NAMES else 'neutral'
    )
    
    # Clean sentiment scores
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['sentiment_score'] = df['sentiment_score'].clip(0, 1)
    
    return df

def trigger_rerun():
    """Programmatically trigger Streamlit rerun"""
    raise RerunException(RerunData())

def session_state_safe(default: Any = None) -> SafeSessionState:
    """
    Safely access Streamlit session state with default value fallback.
    Usage: my_var = session_state_safe('my_key', default_value)
    """
    from streamlit.runtime.state import session_state as _state
    return _state.get(default, default)

def format_timestamp(ts, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert various timestamp formats to consistent string representation"""
    if pd.isna(ts):
        return "N/A"
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime(fmt)
    if isinstance(ts, str):
        return pd.to_datetime(ts).strftime(fmt)
    if isinstance(ts, pd.Timestamp):
        return ts.strftime(fmt)
    return str(ts)