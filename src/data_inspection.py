"""
Data Inspection module for Raw REFIT House 1 data.
Performs:
- Timestamp parsing & ordering checks
- Duplicate timestamp detection
- Sampling interval distribution & irregular interval analysis
- Missing readings and gap analysis
- Extreme and impossible values (negative power, clipping, spikes)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, OUTPUT_DIR


def inspect_raw_file(file_path: Path, max_rows: int = None):
    print(f"\n=======================================================")
    print(f"Inspecting: {file_path.name}")
    print(f"=======================================================")
    
    # Read CSV
    df = pd.read_csv(file_path, nrows=max_rows)
    total_rows = len(df)
    print(f"Total raw rows read: {total_rows:,}")
    
    # 1. Timestamp Parsing & Ordering
    df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    nat_count = int(df['DateTime'].isna().sum())
    
    # Check monotonicity / ordering
    diff_unix = df['Unix'].diff()
    backwards_steps = int((diff_unix < 0).sum())
    zero_diff_steps = int((diff_unix == 0).sum())
    
    # 2. Duplicate Timestamps
    dup_unix_count = int(df.duplicated(subset=['Unix']).sum())
    dup_time_count = int(df.duplicated(subset=['Time']).sum())
    
    # 3. Sampling intervals (dt in seconds)
    valid_dt = diff_unix[diff_unix > 0]
    dt_stats = {
        "mean_seconds": float(valid_dt.mean()) if len(valid_dt) > 0 else 0.0,
        "median_seconds": float(valid_dt.median()) if len(valid_dt) > 0 else 0.0,
        "std_seconds": float(valid_dt.std()) if len(valid_dt) > 0 else 0.0,
        "p50_seconds": float(valid_dt.quantile(0.50)) if len(valid_dt) > 0 else 0.0,
        "p90_seconds": float(valid_dt.quantile(0.90)) if len(valid_dt) > 0 else 0.0,
        "p99_seconds": float(valid_dt.quantile(0.99)) if len(valid_dt) > 0 else 0.0,
        "max_gap_hours": float(valid_dt.max() / 3600.0) if len(valid_dt) > 0 else 0.0,
    }
    
    # Categorize gaps:
    # dt <= 10s: normal polling
    # 10s < dt <= 180s: short delay (can interpolate)
    # dt > 180s (3 min): outage / significant gap (must flag)
    normal_intervals = int((valid_dt <= 10).sum())
    short_gap_intervals = int(((valid_dt > 10) & (valid_dt <= 180)).sum())
    outage_intervals = int((valid_dt > 180).sum())
    
    # 4. Extreme & Impossible Values
    power_cols = [c for c in df.columns if c.startswith('Appliance') or c == 'Aggregate']
    extreme_stats = {}
    for col in power_cols:
        series = pd.to_numeric(df[col], errors='coerce')
        neg_count = int((series < 0).sum())
        nan_count = int(series.isna().sum())
        zero_count = int((series == 0).sum())
        max_val = float(series.max()) if not series.empty else 0.0
        
        # Outlier count: Aggregate > 15000 W or IAM > 4000 W
        threshold = 15000.0 if col == 'Aggregate' else 4000.0
        outliers = int((series > threshold).sum())
        
        extreme_stats[col] = {
            "nan_count": nan_count,
            "zero_count": zero_count,
            "negative_count": neg_count,
            "max_watts": max_val,
            "outliers_above_threshold": outliers,
        }
        
    results = {
        "file": file_path.name,
        "total_rows": total_rows,
        "start_time": str(df['Time'].iloc[0]),
        "end_time": str(df['Time'].iloc[-1]),
        "nat_count": nat_count,
        "backwards_steps": backwards_steps,
        "duplicate_timestamps": dup_unix_count,
        "zero_time_steps": zero_diff_steps,
        "dt_stats": dt_stats,
        "gap_breakdown": {
            "normal_intervals_le_10s": normal_intervals,
            "short_gaps_10s_to_180s": short_gap_intervals,
            "outages_gt_180s": outage_intervals,
        },
        "power_channel_stats": extreme_stats,
    }
    
    print(f"Date range: {results['start_time']} to {results['end_time']}")
    print(f"Duplicate timestamps: {dup_unix_count:,} ({dup_unix_count/total_rows*100:.2f}%)")
    print(f"Backwards timestamp steps: {backwards_steps:,}")
    print(f"Sampling interval (median): {dt_stats['median_seconds']:.1f}s, p99: {dt_stats['p99_seconds']:.1f}s, Max gap: {dt_stats['max_gap_hours']:.1f} hours")
    print(f"Gap breakdown: {normal_intervals:,} normal (<=10s), {short_gap_intervals:,} short gaps, {outage_intervals:,} outages (>3m)")
    
    return results


def run_inspection():
    summary = {}
    for part in ["RAW_House1_Part1.csv", "RAW_House1_Part2.csv"]:
        file_path = RAW_DATA_DIR / part
        if file_path.exists():
            summary[part] = inspect_raw_file(file_path)
            
    out_file = OUTPUT_DIR / "raw_inspection_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nInspection summary saved to {out_file}")
    return summary


if __name__ == "__main__":
    run_inspection()
