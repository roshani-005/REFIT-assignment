"""
NILM Disaggregation Model for Washing Machine
Implements:
1. Feature engineering on 1-minute aggregate power (rolling statistics, lag window, cyclical time)
2. Strict non-leaking chronological train/validation/test split
3. Baseline models (Always-Off baseline and Hour-of-Week Profile baseline)
4. Primary ML model (HistGradientBoostingRegressor / Random Forest)
5. Comprehensive NILM evaluation metrics (Overall MAE, Active MAE, Precision, Recall, F1, RETE)
6. Actual vs Predicted evaluation visualization
7. Automated failure case analysis
"""

from pathlib import Path
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

from src.config import (
    CLEAN_DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    HOUSE_METADATA,
    WM_ACTIVE_THRESHOLD_WATTS,
)

os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".cache")


def create_features(df: pd.DataFrame, agg_col: str = 'Aggregate') -> pd.DataFrame:
    """
    Constructs contextual window features from the aggregate load.
    Avoids target leakage: features use only the Aggregate power series.
    """
    print("Engineering aggregate load window features...")
    agg = df[agg_col]
    
    feats = pd.DataFrame(index=df.index)
    feats['agg_current'] = agg
    
    # Lag and Lead features (receptive field of +/- 15 minutes, total 31 min context)
    for lag in [-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15]:
        col_name = f'agg_lag_{abs(lag)}' if lag > 0 else f'agg_lead_{abs(lag)}'
        feats[col_name] = agg.shift(lag)
        
    # Rolling statistics (centered or backward)
    for w in [5, 15, 30, 60]:
        feats[f'agg_roll_mean_{w}'] = agg.rolling(w, min_periods=1, center=True).mean()
        feats[f'agg_roll_max_{w}'] = agg.rolling(w, min_periods=1, center=True).max()
        feats[f'agg_roll_min_{w}'] = agg.rolling(w, min_periods=1, center=True).min()
        feats[f'agg_roll_std_{w}'] = agg.rolling(w, min_periods=1, center=True).std().fillna(0)
        
    # Rate of change / edges
    feats['agg_diff_1'] = agg.diff(1).fillna(0)
    feats['agg_diff_5'] = agg.diff(5).fillna(0)
    
    # Cyclical temporal features
    hour = df.index.hour
    dow = df.index.dayofweek
    feats['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    feats['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    feats['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    feats['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    # Forward/backward fill small edge NaNs resulting from shifts
    feats = feats.bfill().ffill()
    return feats


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = WM_ACTIVE_THRESHOLD_WATTS):
    """
    Computes comprehensive NILM metrics.
    """
    # Enforce physical non-negativity
    y_pred_clipped = np.clip(y_pred, 0, None)
    # Zero out tiny residual predictions (< 5 W)
    y_pred_clipped[y_pred_clipped < 5.0] = 0.0
    
    # 1. Overall MAE
    mae_overall = float(mean_absolute_error(y_true, y_pred_clipped))
    
    # 2. Active MAE (evaluated strictly when the washing machine is running)
    active_mask = (y_true >= threshold)
    if active_mask.sum() > 0:
        mae_active = float(mean_absolute_error(y_true[active_mask], y_pred_clipped[active_mask]))
    else:
        mae_active = 0.0
        
    # 3. On/Off Classification metrics
    s_true = (y_true >= threshold).astype(int)
    s_pred = (y_pred_clipped >= threshold).astype(int)
    
    prec = float(precision_score(s_true, s_pred, zero_division=0))
    rec = float(recall_score(s_true, s_pred, zero_division=0))
    f1 = float(f1_score(s_true, s_pred, zero_division=0))
    
    # 4. Relative Error in Total Energy (RETE / EER)
    total_true_energy = float(np.sum(y_true) / 60000.0)  # kWh
    total_pred_energy = float(np.sum(y_pred_clipped) / 60000.0) # kWh
    if total_true_energy > 0:
        rete = float(abs(total_pred_energy - total_true_energy) / total_true_energy)
    else:
        rete = 0.0
        
    return {
        "mae_overall_w": round(mae_overall, 2),
        "mae_active_w": round(mae_active, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "total_true_energy_kwh": round(total_true_energy, 2),
        "total_pred_energy_kwh": round(total_pred_energy, 2),
        "rete_energy_error_pct": round(rete * 100.0, 2),
    }


def train_and_evaluate():
    print("\n=======================================================")
    print("Building & Evaluating Appliance Disaggregation Model")
    print("=======================================================")
    
    # 1. Load Cleaned House 1 data
    h1_path = CLEAN_DATA_DIR / "CLEAN_House1.csv"
    meta = HOUSE_METADATA[1]
    wm_col = meta["washing_machine_col"]
    
    print(f"Loading data for House 1 (WM channel: {wm_col})...")
    df = pd.read_csv(h1_path, usecols=['Time', 'Aggregate', wm_col, 'Issues'])
    df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime')
    
    if 'Issues' in df.columns:
        df = df[df['Issues'] == 0]
        
    # 1-minute resampling
    df_1min = df.groupby('DateTime')[['Aggregate', wm_col]].mean().resample('1min').mean()
    df_1min = df_1min.interpolate(method='time', limit=2)
    
    # Drop remaining NaN blocks
    df_clean = df_1min.dropna()
    print(f"Total clean 1-minute intervals: {len(df_clean):,}")
    
    # 2. Strict Chronological Train / Val / Test Split
    # Train: First 70% of timeline
    # Val: Next 15% of timeline
    # Test: Final 15% of timeline
    n = len(df_clean)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)
    
    train_df = df_clean.iloc[:train_idx]
    val_df = df_clean.iloc[train_idx:val_idx]
    test_df = df_clean.iloc[val_idx:]
    
    print(f"Data Split (Chronological, avoids leakage):")
    print(f"  Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df):,} mins, {len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {val_df.index[0]} to {val_df.index[-1]} ({len(val_df):,} mins, {len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df):,} mins, {len(test_df)/n*100:.1f}%)")
    
    # 3. Create Features
    X_train = create_features(train_df, 'Aggregate')
    y_train = train_df[wm_col].values
    
    X_val = create_features(val_df, 'Aggregate')
    y_val = val_df[wm_col].values
    
    X_test = create_features(test_df, 'Aggregate')
    y_test = test_df[wm_col].values
    
    # -------------------------------------------------------------
    # 4. Baselines
    # -------------------------------------------------------------
    print("\n--- Evaluating Baseline Models ---")
    
    # Baseline 1: Always-Off (predicts 0 W)
    y_pred_always_off = np.zeros_like(y_test)
    metrics_always_off = compute_metrics(y_test, y_pred_always_off)
    print(f"Baseline 1 (Always-Off): MAE={metrics_always_off['mae_overall_w']}W | Active MAE={metrics_always_off['mae_active_w']}W | F1={metrics_always_off['f1_score']}")
    
    # Baseline 2: Hour-of-Week Profile (historical mean power per (hour, dow) pair)
    train_profiles = train_df.groupby([train_df.index.dayofweek, train_df.index.hour])[wm_col].mean().to_dict()
    y_pred_profile = np.array([train_profiles.get((t.dayofweek, t.hour), 0.0) for t in test_df.index])
    metrics_profile = compute_metrics(y_test, y_pred_profile)
    print(f"Baseline 2 (Time Profile): MAE={metrics_profile['mae_overall_w']}W | Active MAE={metrics_profile['mae_active_w']}W | F1={metrics_profile['f1_score']} | RETE={metrics_profile['rete_energy_error_pct']}%")
    
    # -------------------------------------------------------------
    # 5. Primary ML Model (HistGradientBoostingRegressor)
    # -------------------------------------------------------------
    print("\n--- Training Primary ML Model (HistGradientBoosting) ---")
    # Using subsample of train data for faster training while retaining all active wash events
    # Ensure active periods are fully represented
    is_active_train = (y_train >= WM_ACTIVE_THRESHOLD_WATTS)
    inactive_train_idx = np.where(~is_active_train)[0]
    # Sample 25% of inactive points to balance dataset and train rapidly
    np.random.seed(42)
    sampled_inactive = np.random.choice(inactive_train_idx, size=int(len(inactive_train_idx) * 0.25), replace=False)
    keep_indices = np.sort(np.concatenate([np.where(is_active_train)[0], sampled_inactive]))
    
    X_train_sub = X_train.iloc[keep_indices]
    y_train_sub = y_train[keep_indices]
    print(f"Training on {len(X_train_sub):,} balanced samples (all {is_active_train.sum():,} active minutes preserved)...")
    
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train_sub, y_train_sub)
    print("Model training complete.")
    
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 0, None)
    # Clean noise below 10W
    y_pred_test[y_pred_test < 10.0] = 0.0
    
    metrics_model = compute_metrics(y_test, y_pred_test)
    print(f"Primary Model (HistGBM): MAE={metrics_model['mae_overall_w']}W | Active MAE={metrics_model['mae_active_w']}W | Precision={metrics_model['precision']} | Recall={metrics_model['recall']} | F1={metrics_model['f1_score']} | RETE={metrics_model['rete_energy_error_pct']}%")
    
    # Comparison Table
    eval_table = {
        "Metric": ["Overall MAE (W)", "Active MAE (W)", "Precision", "Recall", "F1-Score", "True Energy (kWh)", "Pred Energy (kWh)", "Energy Error (RETE %)"],
        "Always-Off Baseline": [
            metrics_always_off["mae_overall_w"],
            metrics_always_off["mae_active_w"],
            metrics_always_off["precision"],
            metrics_always_off["recall"],
            metrics_always_off["f1_score"],
            metrics_always_off["total_true_energy_kwh"],
            metrics_always_off["total_pred_energy_kwh"],
            metrics_always_off["rete_energy_error_pct"],
        ],
        "Profile Baseline": [
            metrics_profile["mae_overall_w"],
            metrics_profile["mae_active_w"],
            metrics_profile["precision"],
            metrics_profile["recall"],
            metrics_profile["f1_score"],
            metrics_profile["total_true_energy_kwh"],
            metrics_profile["total_pred_energy_kwh"],
            metrics_profile["rete_energy_error_pct"],
        ],
        "Primary ML Model (HistGBM)": [
            metrics_model["mae_overall_w"],
            metrics_model["mae_active_w"],
            metrics_model["precision"],
            metrics_model["recall"],
            metrics_model["f1_score"],
            metrics_model["total_true_energy_kwh"],
            metrics_model["total_pred_energy_kwh"],
            metrics_model["rete_energy_error_pct"],
        ]
    }
    df_eval = pd.DataFrame(eval_table)
    print("\n--- Evaluation Comparison Table ---")
    print(df_eval.to_string(index=False))
    
    # Save metrics JSON
    with open(OUTPUT_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump({
            "always_off": metrics_always_off,
            "profile_baseline": metrics_profile,
            "primary_model": metrics_model,
        }, f, indent=2)
        
    # -------------------------------------------------------------
    # 6. Visualization: Actual vs Predicted Plot
    # -------------------------------------------------------------
    plot_actual_vs_predicted(test_df, y_test, y_pred_test, wm_col)
    
    # -------------------------------------------------------------
    # 7. Automated Failure Case Analysis
    # -------------------------------------------------------------
    analyze_failures(test_df, y_test, y_pred_test, wm_col)
    
    return metrics_model, df_eval


def plot_actual_vs_predicted(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, wm_col: str):
    print("Generating Actual vs. Predicted evaluation plot...")
    
    # Select a 5-day window from test set with at least 2 wash cycles
    # Find active timestamps
    active_idx = np.where(y_true > 200.0)[0]
    if len(active_idx) > 0:
        center_time = test_df.index[active_idx[len(active_idx)//2]]
        start_w = center_time - pd.Timedelta(days=2)
        end_w = center_time + pd.Timedelta(days=3)
    else:
        start_w = test_df.index[0]
        end_w = start_w + pd.Timedelta(days=5)
        
    mask = (test_df.index >= start_w) & (test_df.index < end_w)
    plot_times = test_df.index[mask]
    plot_agg = test_df['Aggregate'].values[mask]
    plot_true = y_true[mask]
    plot_pred = y_pred[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.4]})
    
    # Top panel: Aggregate Load
    ax1.plot(plot_times, plot_agg, color='#2c3e50', linewidth=1.1, label='Whole-Home Aggregate Power (W)')
    ax1.set_ylabel("Aggregate (W)", fontsize=11, fontweight='bold')
    ax1.set_title("Evaluation Test Period: Whole-Home Load vs. Washing Machine Disaggregation", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True)
    
    # Bottom panel: True vs Predicted Washing Machine Load
    ax2.plot(plot_times, plot_true, color='#27ae60', linewidth=1.8, label='Ground Truth Washing Machine Power (W)')
    ax2.plot(plot_times, plot_pred, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.9, label='Model Predicted Power (W)')
    ax2.fill_between(plot_times, 0, plot_true, color='#27ae60', alpha=0.2)
    ax2.set_ylabel("Washer Power (W)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Date and Time", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "sec3_actual_vs_predicted.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")


def analyze_failures(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, wm_col: str):
    print("\n--- Analyzing Model Failure Modes ---")
    
    # Failure Mode 1: False Positives (Model predicted wash cycle, but washer was off)
    fp_mask = (y_true < WM_ACTIVE_THRESHOLD_WATTS) & (y_pred >= 200.0)
    fp_count = int(fp_mask.sum())
    
    # Failure Mode 2: False Negatives (Washer was actively running, but model underpredicted or missed)
    fn_mask = (y_true >= 500.0) & (y_pred < 100.0)
    fn_count = int(fn_mask.sum())
    
    print(f"Failure Case 1 (False Positives > 200W): {fp_count} minutes")
    if fp_count > 0:
        fp_agg_mean = float(test_df.loc[fp_mask, 'Aggregate'].mean())
        print(f"  -> Conflation with other appliances: Mean aggregate load during false positives was {fp_agg_mean:.1f} W (e.g. Kettle, Dishwasher, Oven).")
        
    print(f"Failure Case 2 (False Negatives > 500W true): {fn_count} minutes")
    if fn_count > 0:
        fn_agg_mean = float(test_df.loc[fn_mask, 'Aggregate'].mean())
        print(f"  -> Underprediction during high concurrent load: Mean aggregate load during missed wash stages was {fn_agg_mean:.1f} W.")
        
    failure_data = {
        "false_positive_minutes": fp_count,
        "false_negative_minutes": fn_count,
        "fp_agg_mean_watts": float(test_df.loc[fp_mask, 'Aggregate'].mean()) if fp_count > 0 else 0,
        "fn_agg_mean_watts": float(test_df.loc[fn_mask, 'Aggregate'].mean()) if fn_count > 0 else 0,
    }
    with open(OUTPUT_DIR / "failure_analysis.json", "w") as f:
        json.dump(failure_data, f, indent=2)


if __name__ == "__main__":
    train_and_evaluate()
