"""
Exploratory Data Analysis (EDA) of Washing Machine Usage across 4 REFIT Homes
(House 1, House 2, House 3, House 20) using OFFICIAL CLEANED data.

Key updates:
1. Uses shared loader load_cleaned_house() from src.utils
2. Verifies washing machine channel per house against CLEAN_READ_ME_081116.txt
3. Fixes Plot (i) to use a day with a REAL ACTIVE wash cycle (2015-02-23)
4. Fixes Plot (ii) annotation box position to avoid overlapping the legend
5. Accurately describes heating dynamics (pulsed ~3-5 min bursts, not single 12 min)
6. Computes exact diurnal/weekday statistics directly from data
7. Generates cycle-detection sensitivity analysis table (threshold 10/20/50W x gap 5/8/12min)
"""

from pathlib import Path
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    FIGURES_DIR,
    HOUSE_METADATA,
    WM_ACTIVE_THRESHOLD_WATTS,
    WM_CYCLE_MAX_GAP_MINUTES,
    WM_CYCLE_MIN_DURATION_MINUTES,
    WM_CYCLE_MIN_PEAK_WATTS,
    WM_CYCLE_MIN_ENERGY_KWH,
)
from src.utils import load_cleaned_house

os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".cache")


def detect_cycles(series_1min: pd.Series,
                  power_threshold: float = WM_ACTIVE_THRESHOLD_WATTS,
                  max_gap_minutes: int = WM_CYCLE_MAX_GAP_MINUTES,
                  min_duration_minutes: int = WM_CYCLE_MIN_DURATION_MINUTES,
                  min_peak_watts: float = WM_CYCLE_MIN_PEAK_WATTS,
                  min_energy_kwh: float = WM_CYCLE_MIN_ENERGY_KWH):
    """
    Detects washing machine cycles using a finite-state machine with soak-phase bridging.
    """
    blocks = []
    in_cycle = False
    cycle_start = None
    last_active_time = None
    
    times = series_1min.index
    values = series_1min.values
    
    for i in range(len(series_1min)):
        val = values[i]
        t = times[i]
        
        if np.isnan(val):
            if in_cycle and (t - last_active_time).total_seconds() / 60.0 > max_gap_minutes:
                blocks.append((cycle_start, last_active_time))
                in_cycle = False
            continue
            
        if val >= power_threshold:
            if not in_cycle:
                in_cycle = True
                cycle_start = t
            last_active_time = t
        else:
            if in_cycle:
                gap_mins = (t - last_active_time).total_seconds() / 60.0
                if gap_mins > max_gap_minutes:
                    blocks.append((cycle_start, last_active_time))
                    in_cycle = False
                    
    if in_cycle and last_active_time is not None:
        blocks.append((cycle_start, last_active_time))
        
    valid_cycles = []
    for start, end in blocks:
        duration_mins = (end - start).total_seconds() / 60.0 + 1.0
        if duration_mins < min_duration_minutes:
            continue
            
        cycle_slice = series_1min.loc[start:end]
        peak_w = float(cycle_slice.max())
        energy_kwh = float(cycle_slice.sum() / 60000.0)
        
        if peak_w >= min_peak_watts or energy_kwh >= min_energy_kwh:
            valid_cycles.append({
                "start": str(start),
                "end": str(end),
                "duration_minutes": float(duration_mins),
                "peak_watts": float(peak_w),
                "energy_kwh": float(energy_kwh),
                "mean_watts": float(cycle_slice.mean()),
                "hour": int(start.hour),
                "day_of_week": int(start.dayofweek), # 0=Mon, 6=Sun
            })
            
    return valid_cycles


def compute_cycle_sensitivity(house_dfs):
    """
    Computes cycle count sensitivity across power thresholds (10, 20, 50 W)
    and bridging gaps (5, 8, 12 min).
    """
    print("\nRunning Cycle Detection Sensitivity Analysis...")
    thresholds = [10.0, 20.0, 50.0]
    gaps = [5, 8, 12]
    sensitivity_results = {}
    
    for hid, df in house_dfs.items():
        wm_col = HOUSE_METADATA[hid]["washing_machine_col"]
        series = df[wm_col]
        h_res = {}
        for th in thresholds:
            for g in gaps:
                key = f"th_{int(th)}W_gap_{g}m"
                detected = detect_cycles(
                    series,
                    power_threshold=th,
                    max_gap_minutes=g,
                    min_duration_minutes=WM_CYCLE_MIN_DURATION_MINUTES,
                    min_peak_watts=WM_CYCLE_MIN_PEAK_WATTS,
                    min_energy_kwh=WM_CYCLE_MIN_ENERGY_KWH
                )
                h_res[key] = len(detected)
        sensitivity_results[f"House_{hid}"] = h_res
        
    return sensitivity_results


def run_eda():
    houses = [1, 2, 3, 20]
    eda_summary = {}
    house_dfs = {}
    house_cycles = {}
    
    for hid in houses:
        meta = HOUSE_METADATA[hid]
        df_1min, wm_col = load_cleaned_house(hid)
        house_dfs[hid] = df_1min
        
        total_mins = len(df_1min)
        start_date = str(df_1min.index[0])
        end_date = str(df_1min.index[-1])
        days = (df_1min.index[-1] - df_1min.index[0]).total_seconds() / 86400.0
        
        # Missing minutes (original + unfillable)
        missing_mins = int(df_1min[wm_col].isna().sum())
        missing_pct = float(missing_mins / total_mins * 100.0)
        
        # Zero / Off / Standby time (Power < 20 W)
        valid_series = df_1min[wm_col].dropna()
        off_mins = int((valid_series < WM_ACTIVE_THRESHOLD_WATTS).sum())
        off_pct = float(off_mins / len(valid_series) * 100.0) if len(valid_series) > 0 else 0.0
        
        # Detect cycles
        print(f"Detecting washing machine cycles for House {hid} (Ch: {wm_col})...")
        cycles = detect_cycles(df_1min[wm_col])
        house_cycles[hid] = cycles
        num_cycles = len(cycles)
        
        durations = [c['duration_minutes'] for c in cycles]
        energies = [c['energy_kwh'] for c in cycles]
        peaks = [c['peak_watts'] for c in cycles]
        
        eda_summary[f"House_{hid}"] = {
            "house_id": hid,
            "channel": wm_col,
            "occupancy": meta["occupancy"],
            "type": meta["type"],
            "wm_model": meta["wm_model"],
            "date_range": f"{start_date} to {end_date}",
            "days_covered": round(days, 1),
            "total_minutes": total_mins,
            "missing_minutes": missing_mins,
            "missing_pct": round(missing_pct, 2),
            "zero_off_minutes": off_mins,
            "zero_off_pct": round(off_pct, 2),
            "detected_cycles": num_cycles,
            "cycles_per_week": round(num_cycles / (days / 7.0), 2) if days > 0 else 0,
            "mean_duration_mins": round(float(np.mean(durations)), 1) if durations else 0,
            "median_duration_mins": round(float(np.median(durations)), 1) if durations else 0,
            "mean_energy_kwh": round(float(np.mean(energies)), 3) if energies else 0,
            "median_energy_kwh": round(float(np.median(energies)), 3) if energies else 0,
            "mean_peak_watts": round(float(np.mean(peaks)), 1) if peaks else 0,
            "total_washing_energy_kwh": round(float(np.sum(energies)), 1) if energies else 0,
        }
        
    # Aggregate statistics across all 4 houses
    all_cycles = []
    for hid in houses:
        for c in house_cycles[hid]:
            c_copy = dict(c)
            c_copy['house_id'] = hid
            all_cycles.append(c_copy)
            
    df_all_cycles = pd.DataFrame(all_cycles)
    df_all_cycles['start_dt'] = pd.to_datetime(df_all_cycles['start'])
    
    # Exact weekday counts and shares
    dow_counts = df_all_cycles['start_dt'].dt.day_name().value_counts().to_dict()
    dow_shares = (df_all_cycles['start_dt'].dt.day_name().value_counts(normalize=True) * 100.0).round(2).to_dict()
    
    # Exact hour counts and shares
    hour_counts = df_all_cycles['hour'].value_counts().to_dict()
    morning_count = int(df_all_cycles['hour'].between(8, 11).sum())
    morning_share = round(float(morning_count / len(df_all_cycles) * 100.0), 2)
    
    # Overnight washing in House 1 (00:00 to 06:00)
    h1_cycles = [c for c in all_cycles if c['house_id'] == 1]
    h1_overnight = [c for c in h1_cycles if 0 <= c['hour'] < 6]
    h1_overnight_share = round(float(len(h1_overnight) / len(h1_cycles) * 100.0), 2) if h1_cycles else 0.0
    
    eda_summary["aggregate_patterns"] = {
        "total_cycles_4_houses": len(df_all_cycles),
        "weekday_counts": dow_counts,
        "weekday_percentages": dow_shares,
        "highest_day": max(dow_counts, key=dow_counts.get),
        "lowest_day": min(dow_counts, key=dow_counts.get),
        "morning_08_to_12_count": morning_count,
        "morning_08_to_12_pct": morning_share,
        "house1_overnight_00_to_06_count": len(h1_overnight),
        "house1_overnight_00_to_06_pct": h1_overnight_share,
    }
    
    # Sensitivity analysis
    eda_summary["sensitivity_analysis"] = compute_cycle_sensitivity(house_dfs)
    
    # Save JSON summary
    summary_path = OUTPUT_DIR / "washing_machine_eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\nEDA summary saved to {summary_path}")
    
    # Generate Visualizations
    generate_all_eda_plots(house_dfs, house_cycles, eda_summary)
    
    return eda_summary, house_dfs, house_cycles


def generate_all_eda_plots(house_dfs, house_cycles, eda_summary):
    print("\n--- Generating Clean Figures for Section 2 ---")
    
    # -------------------------------------------------------------
    # Plot (i): 24-hour aggregate + washing machine trace
    # Fixed: Uses 2015-02-23 (REAL active wash day in House 1, date in title)
    # -------------------------------------------------------------
    print("Generating Plot (i): 24-hour trace with genuine active wash day...")
    h1_df = house_dfs[1]
    wm_col1 = HOUSE_METADATA[1]["washing_machine_col"]
    
    # Active day: 2015-02-23
    day_start = pd.Timestamp("2015-02-23 00:00:00")
    day_end = pd.Timestamp("2015-02-23 23:59:00")
    slice_24h = h1_df.loc[day_start:day_end]
    
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(slice_24h.index, slice_24h['Aggregate'], label='Household Aggregate Load (W)', color='#1f77b4', linewidth=1.2, alpha=0.85)
    ax.plot(slice_24h.index, slice_24h[wm_col1], label='Washing Machine (Beko WMC6140) (W)', color='#d62728', linewidth=1.6)
    ax.fill_between(slice_24h.index, 0, slice_24h[wm_col1], color='#d62728', alpha=0.25, label='Washing Machine Active Energy')
    
    ax.set_title("Plot (i): Representative 24-Hour Electricity Load Trace (House 1 - 23 Feb 2015)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time of Day", fontsize=11, fontweight='bold')
    ax.set_ylabel("Active Power (Watts)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "sec2_24h_trace.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # -------------------------------------------------------------
    # Plot (ii): Zoomed-in detected washing-machine cycle
    # Fixed: Repositioned annotation to avoid overlapping legend
    # Accurately describes pulsed heating (~3-5 mins each)
    # -------------------------------------------------------------
    print("Generating Plot (ii): Zoomed detected cycle with clean annotation...")
    h1_cycles = house_cycles[1]
    chosen_cycle = None
    for c in h1_cycles:
        if 0.6 <= c['energy_kwh'] <= 1.4 and 60 <= c['duration_minutes'] <= 140:
            chosen_cycle = c
            break
    if chosen_cycle is None:
        chosen_cycle = h1_cycles[0]
        
    c_start = pd.Timestamp(chosen_cycle['start']) - pd.Timedelta(minutes=10)
    c_end = pd.Timestamp(chosen_cycle['end']) + pd.Timedelta(minutes=10)
    cycle_slice = h1_df.loc[c_start:c_end]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cycle_slice.index, cycle_slice[wm_col1], color='#d62728', linewidth=2.0, label='Washing Machine Load (W)')
    
    det_start = pd.Timestamp(chosen_cycle['start'])
    det_end = pd.Timestamp(chosen_cycle['end'])
    ax.axvline(det_start, color='green', linestyle='--', linewidth=1.8, label=f'Cycle Start ({det_start.strftime("%H:%M")})')
    ax.axvline(det_end, color='purple', linestyle='--', linewidth=1.8, label=f'Cycle End ({det_end.strftime("%H:%M")})')
    
    # Position annotation in the center-left to strictly avoid the upper-right legend
    peak_time = cycle_slice[wm_col1].idxmax()
    peak_val = cycle_slice[wm_col1].max()
    ax.annotate('Pulsed Water Heating\n(~2200-2400 W, ~3-5 min bursts)',
                xy=(peak_time, peak_val),
                xytext=(c_start + pd.Timedelta(minutes=5), peak_val * 0.75),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", alpha=0.85))
                
    ax.set_title(f"Plot (ii): Detected Wash Cycle Anatomy (House 1 - {det_start.strftime('%d %b %Y')})\nDuration: {chosen_cycle['duration_minutes']:.0f} min | Peak: {chosen_cycle['peak_watts']:.0f} W | Energy: {chosen_cycle['energy_kwh']:.2f} kWh",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Time", fontsize=11, fontweight='bold')
    ax.set_ylabel("Power (Watts)", fontsize=11, fontweight='bold')
    ax.set_ylim(-50, max(peak_val * 1.25, 2600))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "sec2_detected_cycles.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # -------------------------------------------------------------
    # Plot (iii): Cross-House Comparison
    # -------------------------------------------------------------
    print("Generating Plot (iii): Cross-house comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    house_labels = [f"House {hid}\n({HOUSE_METADATA[hid]['occupancy']} occ)" for hid in [1, 2, 3, 20]]
    dur_data = [[c['duration_minutes'] for c in house_cycles[hid]] for hid in [1, 2, 3, 20]]
    nrg_data = [[c['energy_kwh'] for c in house_cycles[hid]] for hid in [1, 2, 3, 20]]
    peak_data = [[c['peak_watts'] for c in house_cycles[hid]] for hid in [1, 2, 3, 20]]
    box_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    bp1 = axes[0].boxplot(nrg_data, tick_labels=house_labels, patch_artist=True, showmeans=True)
    for patch, col in zip(bp1['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    axes[0].set_title("Energy per Cycle (kWh)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Energy (kWh)", fontsize=11, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    
    bp2 = axes[1].boxplot(dur_data, tick_labels=house_labels, patch_artist=True, showmeans=True)
    for patch, col in zip(bp2['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    axes[1].set_title("Cycle Duration (Minutes)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Minutes", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    
    bp3 = axes[2].boxplot(peak_data, tick_labels=house_labels, patch_artist=True, showmeans=True)
    for patch, col in zip(bp3['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    axes[2].set_title("Peak Power per Cycle (Watts)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Watts", fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.4)
    
    plt.suptitle("Plot (iii): Cross-Household Washing Machine Cycle Characteristics Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "sec2_cross_house_comparison.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # Plot (iv): Temporal Patterns (Hour of day & Day of week)
    # -------------------------------------------------------------
    print("Generating Plot (iv): Temporal patterns...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_cycles = []
    for hid in [1, 2, 3, 20]:
        for c in house_cycles[hid]:
            c_copy = dict(c)
            c_copy['house'] = f"House {hid}"
            all_cycles.append(c_copy)
            
    hours = np.arange(24)
    hour_counts = {f"House {hid}": [0]*24 for hid in [1, 2, 3, 20]}
    for c in all_cycles:
        hour_counts[c['house']][c['hour']] += 1
        
    bottom = np.zeros(24)
    for i, hid in enumerate([1, 2, 3, 20]):
        h_name = f"House {hid}"
        axes[0].bar(hours, hour_counts[h_name], bottom=bottom, label=h_name, color=box_colors[i], alpha=0.8)
        bottom += np.array(hour_counts[h_name])
        
    axes[0].set_title("Usage Pattern by Hour of Day", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hour of Day (0-23)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Total Cycles Started", fontsize=11, fontweight='bold')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(loc='upper left', frameon=True)
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = {f"House {hid}": [0]*7 for hid in [1, 2, 3, 20]}
    for c in all_cycles:
        dow_counts[c['house']][c['day_of_week']] += 1
        
    bottom_dow = np.zeros(7)
    for i, hid in enumerate([1, 2, 3, 20]):
        h_name = f"House {hid}"
        axes[1].bar(dow_names, dow_counts[h_name], bottom=bottom_dow, label=h_name, color=box_colors[i], alpha=0.8)
        bottom_dow += np.array(dow_counts[h_name])
        
    axes[1].set_title("Usage Pattern by Day of Week", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Day of Week", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Total Cycles Started", fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper right', frameon=True)
    
    plt.suptitle("Washing Machine Temporal Usage Patterns Across 4 REFIT Homes", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4_path = FIGURES_DIR / "sec2_temporal_patterns.png"
    plt.savefig(fig4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig4_path}")


if __name__ == "__main__":
    run_eda()
