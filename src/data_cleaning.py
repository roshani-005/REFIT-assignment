"""
Raw Data Cleaning Pipeline for REFIT House 1
Implements:
1. Merging Part1 and Part2 with priority to Part2 on overlap
2. Keeping a pristine unclipped copy for Before/After plotting
3. Outlier / Sensor error handling (clipping impossible spikes on working copy)
4. Sorting readings chronologically
5. Deduplicating timestamps using stated rule (mean aggregation of concurrent seconds)
6. 1-minute uniform resampling
7. Short gap interpolation (runs <= 3 min only) and outage flagging (runs > 3 min)
   via src.utils.fill_short_gaps and src.utils.compute_outage_mask
8. Before/After cleaning visualization with TRULY RAW unclipped data
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
from src.utils import fill_short_gaps, compute_outage_mask

# Set matplotlib cache to writable directory
os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".cache")


def clean_raw_house1():
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
    
    # Transition point: use Part1 before Part2 start, Part2 thereafter
    transition_date = df2['DateTime'].min()
    print(f"Transition point: Using Part1 before {transition_date}, and Part2 from {transition_date} onwards.")
    df_raw_merged = pd.concat([df1[df1['DateTime'] < transition_date], df2], ignore_index=True)
    del df1, df2
    
    # Sort chronologically
    df_raw_merged = df_raw_merged.sort_values(by='DateTime').reset_index(drop=True)
    raw_len = len(df_raw_merged)
    
    channels = ['Aggregate'] + [f'Appliance{i}' for i in range(1, 10)]
    
    # Keep an UNFILTERED pristine copy for the "Before" plot slice
    plot_start = pd.Timestamp("2013-11-15 00:00:00")
    plot_end = pd.Timestamp("2013-11-18 00:00:00")
    raw_window_unfiltered = df_raw_merged[
        (df_raw_merged['DateTime'] >= plot_start) & (df_raw_merged['DateTime'] < plot_end)
    ].copy()
    
    # Work on a copy for cleaning
    df_cleaned = df_raw_merged.copy()
    for ch in channels:
        df_cleaned[ch] = pd.to_numeric(df_cleaned[ch], errors='coerce')
        # Negative values are impossible for power sensors -> set to NaN
        df_cleaned.loc[df_cleaned[ch] < 0, ch] = np.nan
        # Extreme sensor overflow / electrical impossibility -> set to NaN
        limit = POWER_UPPER_LIMIT_AGG if ch == 'Aggregate' else POWER_UPPER_LIMIT_IAM
        df_cleaned.loc[df_cleaned[ch] > limit, ch] = np.nan

    # Deduplicate timestamps using stated rule:
    # Stated Rule: Average concurrent readings within the exact same second to preserve energy balance
    print("Deduplicating timestamps by averaging concurrent readings per second...")
    df_dedup = df_cleaned.groupby('DateTime')[channels].mean()
    dedup_len = len(df_dedup)
    print(f"Deduplication: {raw_len:,} rows -> {dedup_len:,} unique seconds ({raw_len - dedup_len:,} duplicates resolved).")
    
    # Resample to 1-minute intervals
    print(f"Resampling to uniform {RESAMPLE_INTERVAL} grid...")
    df_1min = df_dedup.resample(RESAMPLE_INTERVAL).mean()
    total_minutes = len(df_1min)
    
    # Apply robust gap policy:
    # 1. Compute outage mask from ORIGINAL resampled series (runs > 3 min)
    # 2. Fill ONLY runs <= 3 min using linear interpolation
    # 3. Runs > 3 min strictly remain NaN
    print("Applying gap policy: fill only runs <= 3 min; leave runs > 3 min as NaN...")
    outage_mask = compute_outage_mask(df_1min['Aggregate'], max_gap=MAX_INTERPOLATION_MINUTES)
    
    for ch in channels:
        df_1min[ch] = fill_short_gaps(df_1min[ch], max_gap=MAX_INTERPOLATION_MINUTES)
        
    df_1min['is_outage'] = outage_mask
    outage_count = int(outage_mask.sum())
    
    print(f"Total 1-minute intervals: {total_minutes:,}")
    print(f"Flagged outages (> {MAX_INTERPOLATION_MINUTES} min): {outage_count:,} minutes ({outage_count/total_minutes*100:.2f}%)")
    
    # Save cleaned 1-minute parquet
    out_parquet = OUTPUT_DIR / "House1_cleaned_1min.parquet"
    df_1min.to_parquet(out_parquet)
    print(f"Cleaned 1-minute data saved to {out_parquet}")
    
    # Plot Before (TRULY RAW UNFILTERED) vs After (Cleaned 1-minute)
    cleaned_window = df_1min.loc[plot_start:plot_end]
    plot_before_after(raw_window_unfiltered, cleaned_window)
    
    return {
        "raw_total_rows": raw_len,
        "dedup_rows": dedup_len,
        "total_1min_intervals": total_minutes,
        "outage_minutes": outage_count,
        "outage_pct": round(float(outage_count / total_minutes * 100.0), 2),
    }


def plot_before_after(raw_df, clean_df):
    print("Generating Before vs After cleaning visualization (using TRULY RAW unclipped data for 'Before')...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Channels: Aggregate, Appliance1 (Fridge), Appliance5 (Washing Machine)
    plot_configs = [
        ("Aggregate", "Aggregate Power (W)", "navy"),
        ("Appliance1", "Appliance 1: Fridge (W)", "darkgreen"),
        ("Appliance5", "Appliance 5: Washing Machine (W)", "crimson"),
    ]
    
    for idx, (col, title, color) in enumerate(plot_configs):
        ax = axes[idx]
        
        # Raw data (truly raw irregular readings)
        raw_vals = pd.to_numeric(raw_df[col], errors='coerce')
        ax.plot(
            raw_df['DateTime'],
            raw_vals,
            color='gray',
            alpha=0.55,
            linewidth=0.8,
            label='Raw irregular (~7s, unclipped)',
        )
        
        # Cleaned data (uniform 1-minute resampled, short gaps filled)
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
