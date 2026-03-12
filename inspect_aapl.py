"""Quick AAPL parquet inspector — run from K:\_DEV_MVP_2026\Market_Hawk_3"""
import sys, pandas as pd
from backtesting.data_loader import HistoricalDataLoader

loader = HistoricalDataLoader()
loader.scan_available_data()

# Find the AAPL entry
for key, info in loader._index.items():
    if "AAPL" in key.upper():
        print(f"Key: {key}")
        print(f"Path: {info['path']}")
        print(f"Format: {info['format']}, Size: {info['size_mb']} MB")
        
        df = pd.read_parquet(info['path'])
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index name: {df.index.name}, dtype: {df.index.dtype}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        print(f"\nLast 3 rows:")
        print(df.tail(3).to_string())
        print(f"\nColumn dtypes:")
        print(df.dtypes)
        break
