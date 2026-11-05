"""
Raw Data Cleaning Pipeline for REFIT House 1
Implements:
1. Merging Part1 and Part2 with priority to Part2 on overlap
2. Sorting readings chronologically
3. Deduplicating timestamps using stated rule (mean aggregation)
4. Outlier / Sensor error handling (clipping impossible spikes)
5. 1-minute uniform resampling
6. Short gap interpolation (<= 3 min) and outage flagging (> 3 min)
7. Before/After cleaning visualization
"""

from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    RESAMPLE_INTERVAL,
    MAX_INTERPOLATION_MINUTES,
    POWER_UPPER_LIMIT_AGG,
    POWER_UPPER_LIMIT_IAM,
)

# Set matplotlib cache to writable directory
os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".cache")


def clean_raw_house1(sample_days: int = None):
    print("\n--- Starting Raw House 1 Cleaning Pipeline ---")
    
    # 1. Load Part1 and Part2
    p1_path = RAW_DATA_DIR / "RAW_House1_Part1.csv"
    p2_path = RAW_DATA_DIR / "RAW_House1_Part2.csv"
    
    print("Loading RAW_House1_Part1...")
    df1 = pd.read_csv(p1_path)
    df1['DateTime'] = pd.to_datetime(df1['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    print("Loading RAW_House1_Part2...")
    df2 = pd.read_csv(p2_path)
    df2['DateTime'] = pd.to_datetime(df2['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Filter out NaT timestamps
    df1 = df1.dropna(subset=['DateTime'])
    df2 = df2.dropna(subset=['DateTime'])
    
    # In overlap period (June 27 2014 to Oct 1 2014), Part2 has NaNs for missing sensors, which is cleaner
    transition_date = df2['DateTime'].min()
    print(f"Transition point: Using Part1 before {transition_date}, and Part2 from {transition_date} onwards.")
    df_raw = pd.concat([df1[df1['DateTime'] < transition_date], df2], ignore_index=True)
    del df1, df2
    
    # 2. Sort chronologically
    print("Sorting readings chronologically...")
    df_raw = df_raw.sort_values(by='DateTime').reset_index(drop=True)
    
    # Power columns
    channels = ['Aggregate'] + [f'Appliance{i}' for i in range(1, 10)]
    for ch in channels:
        df_raw[ch] = pd.to_numeric(df_raw[ch], errors='coerce')
        # Negative values are impossible for power sensors -> set to NaN
        df_raw.loc[df_raw[ch] < 0, ch] = np.nan
        # Extreme sensor overflow / electrical impossibility
        limit = POWER_UPPER_LIMIT_AGG if ch == 'Aggregate' else POWER_UPPER_LIMIT_IAM
        df_raw.loc[df_raw[ch] > limit, ch] = np.nan

    # 3. Deduplicate timestamps using stated rule:
    # Stated Rule: Average concurrent readings within the exact same second to preserve energy balance
    print("Deduplicating timestamps by averaging concurrent readings per second...")
    raw_len = len(df_raw)
    df_dedup = df_raw.groupby('DateTime')[channels].mean()
    dedup_len = len(df_dedup)
    print(f"Deduplication reduced {raw_len:,} rows to {dedup_len:,} unique 1-second timestamps ({raw_len - dedup_len:,} duplicates removed).")
    
    # 4. Resample to 1-minute intervals
    print(f"Resampling to uniform {RESAMPLE_INTERVAL} grid...")
    df_1min = df_dedup.resample(RESAMPLE_INTERVAL).mean()
    
    # 5. Handle missing gaps:
    # Short gaps (<= 3 mins): linearly interpolate
    # Outages (> 3 mins): flag as outage, leave as NaN
    print("Applying gap policy: linear interpolation for <= 3 min gaps, outage flagging for > 3 min...")
    df_interpolated = df_1min.interpolate(method='time', limit=MAX_INTERPOLATION_MINUTES)
    
    # Boolean mask for outages (unfilled gaps)
    is_outage = df_interpolated['Aggregate'].isna()
    outage_count = is_outage.sum()
    total_minutes = len(df_1min)
    print(f"Total 1-minute intervals: {total_minutes:,}")
    print(f"Flagged outages (> {MAX_INTERPOLATION_MINUTES} min): {outage_count:,} minutes ({outage_count/total_minutes*100:.2f}%)")
    
    # Save a slice or full parquet for efficiency
    out_parquet = OUTPUT_DIR / "House1_cleaned_1min.parquet"
    df_interpolated['is_outage'] = is_outage
    df_interpolated.to_parquet(out_parquet)
    print(f"Cleaned 1-minute data saved to {out_parquet}")
    
    # 6. Generate Before vs After Plot
    # Select a representative 3-day window containing active use, short gap, and transition
    plot_start = pd.Timestamp("2013-11-15 00:00:00")
    plot_end = pd.Timestamp("2013-11-18 00:00:00")
    
    raw_window = df_raw[(df_raw['DateTime'] >= plot_start) & (df_raw['DateTime'] < plot_end)]
    cleaned_window = df_interpolated.loc[plot_start:plot_end]
    
    plot_before_after(raw_window, cleaned_window)
    
    return {
        "raw_total_rows": raw_len,
        "dedup_rows": dedup_len,
        "total_1min_intervals": total_minutes,
        "outage_minutes": int(outage_count),
        "outage_pct": float(outage_count / total_minutes * 100),
    }


def plot_before_after(raw_df, clean_df):
    print("Generating Before vs After cleaning visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Channels to plot: Aggregate, Appliance1 (Fridge), Appliance5 (Washing Machine)
    plot_configs = [
        ("Aggregate", "Aggregate Power (W)", "navy"),
        ("Appliance1", "Appliance 1: Fridge (W)", "darkgreen"),
        ("Appliance5", "Appliance 5: Washing Machine (W)", "crimson"),
    ]
    
    for idx, (col, title, color) in enumerate(plot_configs):
        ax = axes[idx]
        
        # Raw data (scatter / thin line showing irregular intervals)
        ax.plot(
            raw_df['DateTime'],
            raw_df[col],
            color='gray',
            alpha=0.5,
            linewidth=0.8,
            label='Raw irregular (~7s)',
        )
        
        # Cleaned data (uniform 1-minute resampled)
        ax.plot(
            clean_df.index,
            clean_df[col],
            color=color,
            alpha=0.9,
            linewidth=1.3,
            label='Cleaned 1-min resampled',
        )
        
        ax.set_ylabel("Power (W)", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', frameon=True)
        
    axes[-1].set_xlabel("Date and Time", fontsize=11, fontweight='bold')
    plt.suptitle("House 1: Power Readings Before vs. After Cleaning & 1-Minute Resampling", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / "sec1_raw_vs_cleaned.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    clean_raw_house1()
