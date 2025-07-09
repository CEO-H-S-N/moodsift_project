import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from datetime import datetime
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from typing import Union, Optional, Dict, List

class DataStorage:
    """Handles persistent data storage and retrieval"""
    
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(self, 
                     df: pd.DataFrame, 
                     source: str, 
                     timestamp: Optional[datetime] = None) -> Path:
        """
        Save raw collected data with automatic timestamping
        Args:
            df: DataFrame containing raw data
            source: Data source identifier (e.g., 'reddit', 'twitter')
            timestamp: Optional specific timestamp
        Returns:
            Path to saved file
        """
        ts = timestamp or datetime.now()
        filename = f"{source}_raw_{ts.strftime('%Y%m%d_%H%M%S')}.parquet"
        filepath = self.raw_dir / filename
        df.to_parquet(filepath, engine='pyarrow')
        return filepath
    
    def save_processed_data(self, 
                           df: pd.DataFrame, 
                           source: str,
                           analysis_type: str = 'sentiment') -> Path:
        """
        Save processed/analyzed data with versioning
        Args:
            df: Processed DataFrame
            source: Data source identifier
            analysis_type: Type of analysis performed
        Returns:
            Path to saved file
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{source}_{analysis_type}_{ts}.parquet"
        filepath = self.processed_dir / filename
        
        # Enhanced Parquet writing with schema preservation
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            filepath,
            compression='snappy',
            coerce_timestamps='ms',
            allow_truncated_timestamps=True
        )
        return filepath
    
    def load_latest_data(self, 
                        source: str, 
                        processed: bool = True) -> Optional[pd.DataFrame]:
        """
        Load most recent data file for a given source
        Args:
            source: Data source identifier
            processed: Whether to load processed or raw data
        Returns:
            DataFrame if found, else None
        """
        directory = self.processed_dir if processed else self.raw_dir
        pattern = f"{source}_*.parquet"
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        
        if files:
            return pd.read_parquet(files[0])
        return None
    
    def batch_load_data(self, 
                       time_range: Dict[str, datetime] = None, 
                       sources: List[str] = None) -> pd.DataFrame:
        """
        Load multiple data files matching criteria
        Args:
            time_range: {'start': datetime, 'end': datetime}
            sources: List of source identifiers to include
        Returns:
            Concatenated DataFrame
        """
        frames = []
        search_dir = self.processed_dir if time_range else self.raw_dir
        
        for file in search_dir.glob("*.parquet"):
            file_ts = datetime.strptime(file.stem.split('_')[-1], '%Y%m%d_%H%M%S')
            
            # Filter by time range if specified
            if time_range and not (time_range['start'] <= file_ts <= time_range['end']):
                continue
            
            # Filter by source if specified
            if sources and file.stem.split('_')[0] not in sources:
                continue
            
            df = pd.read_parquet(file)
            df['data_source'] = file.stem.split('_')[0]
            df['collection_time'] = file_ts
            frames.append(df)
        
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def get_available_sources(self) -> List[str]:
        """List all unique data sources available"""
        raw_sources = {f.stem.split('_')[0] for f in self.raw_dir.glob("*_raw_*.parquet")}
        processed_sources = {f.stem.split('_')[0] for f in self.processed_dir.glob("*_sentiment_*.parquet")}
        return sorted(raw_sources.union(processed_sources))