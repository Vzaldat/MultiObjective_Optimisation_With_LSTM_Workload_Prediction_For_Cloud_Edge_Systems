import pandas as pd
import numpy as np

# Load up the container usage data
# columns - ts, instance_id, cpu_util, mem_util, disk_util, load1, load5, load15, avg_cpi, avg_mpki

df = pd.read_csv('alibaba-trace-2017/container_usage.csv', names=['ts', 'instance_id', 'cpu_util', 'mem_util', 'disk_util', 'load1', 'load5', 'load15', 'avg_cpi', 'avg_mpki', 'max_cpi', 'max_mpki'])


features = ['cpu_util', 'avg_cpi', 'avg_mpki']

# 1. Force columns to numeric, converting string errors/spaces to NaN
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 2. Forward-fill missing values chronologically, then backward-fill any remnants
    df[features] = df[features].ffill().bfill()
    
    # 3. Clip infinite values or massive anomalies that cause exploding gradients
    df[features] = np.clip(df[features].values, a_min=0.0, a_max=100000.0)
    
    # 4. Double check to ensure absolutely zero NaNs remain
    if df[features].isna().any().any():
        df = df.dropna(subset=features)
        
    # Reset indices to prevent mapping alignment drops
    df = df.reset_index(drop=True)

print(df[features].isna().sum())                        # NaN counts
print(np.isinf(df[features].values).sum(axis=0))        # inf counts
print(df[features].describe())  

df = df.dropna()

# Group by the timestamp (ts) and instance_id to find total load at any given second

workload_time_series = df.groupby(['ts', 'instance_id']).agg({
    'cpu_util' : 'mean', # Average CPU utilisation percentage across the cluster
    'avg_cpi' : 'mean', # Average CPI (Proxy for compute efficiency)
    'avg_mpki' : 'mean' # Average cache misses (Memory subsystem stress)
}).reset_index()

workload_time_series.to_csv('workload_time_series.csv', index=False)