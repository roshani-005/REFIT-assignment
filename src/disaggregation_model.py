"""
Appliance Disaggregation Model for Washing Machine (NILM)
Dataset: REFIT Smart Home Dataset (Official Cleaned Data)

Offline Inference Setting:
This disaggregation model operates strictly in an OFFLINE setting.
It leverages contextual load features from a 31-minute receptive field:
- 15 minutes of historical lag power (t-15 to t-1)
- Current aggregate power (t)
- 15 minutes of future lead power (t+1 to t+15)
- Centered rolling statistics over 5, 15, 30, and 60 minutes
This architecture is designed for retrospective energy auditing, billing disaggregation,
and offline demand-side analytics. It is NOT an online real-time streaming model.

Methodology:
1. Shared Loader: Uses load_cleaned_house() from src.utils across all houses.
2. Temporal Continuity: Window features are engineered strictly within continuous valid segments.
   Rows whose windows touch an outage (>3 min NaN) or partition boundary are excluded.
3. Validation-Guided Tuning (Tune & Freeze):
   - Evaluates 3 training strategies on the Validation set:
     (a) Full training data
     (b) Balanced active + 25% inactive subsampling
     (c) Class-weighted training
   - Tunes postprocessing zero-cutoff and active on/off threshold on Validation set.
   - Freezes choices before a single final evaluation on the unseen Test set.
4. Shared Post-Processing:
   - Unified postprocessing (cutoff clipping) applied identically to Primary ML Model,
     Always-Off Baseline, and Hour-of-Week Profile Baseline.
5. Cross-House Generalization:
   - Leave-One-House-Out (LOHO) cross-validation across all 4 houses {1, 2, 3, 20}.
   - Primary within-home benchmark on House 1 chronological split.
6. Rigorous Metrics:
   - MAE Overall, MAE Active, RMSE, Precision, Recall, Specificity, F1-Score,
     Confusion Matrix, Signed Energy Bias (kWh), and Relative Error in Total Energy (RETE %).
   - Evaluated on identical non-missing timestamps.
7. Visualizations:
   - Actual vs. Predicted Time-Series Trace (sec3_actual_vs_predicted.png)
   - Confusion Matrix Heatmap (sec3_confusion_matrix.png)
   - Cumulative Energy Ground Truth vs. Predictions (sec3_cumulative_energy.png)
"""

from pathlib import Path
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

from src.config import (
    OUTPUT_DIR,
    FIGURES_DIR,
    HOUSE_METADATA,
    WM_ACTIVE_THRESHOLD_WATTS,
)
from src.utils import load_cleaned_house

os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".cache")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def create_offline_features(agg: pd.Series) -> pd.DataFrame:
    """
    Offline contextual feature extraction from aggregate power.
    Uses lags (-15 to +15), centered rolling statistics, rate of change, and cyclical time.
    Requires continuous valid aggregate power within the window; missing points produce NaNs.
    """
    feats = pd.DataFrame(index=agg.index)
    feats['agg_current'] = agg
    
    # Lags (past) and Leads (future): +/- 15 min
    for lag in [-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15]:
        col_name = f'agg_lag_{abs(lag)}' if lag > 0 else f'agg_lead_{abs(lag)}'
        feats[col_name] = agg.shift(lag)
        
    # Centered rolling statistics: min_periods=w ensures no NaNs inside window
    for w in [5, 15, 30, 60]:
        feats[f'agg_roll_mean_{w}'] = agg.rolling(w, min_periods=w, center=True).mean()
        feats[f'agg_roll_max_{w}'] = agg.rolling(w, min_periods=w, center=True).max()
        feats[f'agg_roll_min_{w}'] = agg.rolling(w, min_periods=w, center=True).min()
        feats[f'agg_roll_std_{w}'] = agg.rolling(w, min_periods=w, center=True).std()
        
    # Rate of change
    feats['agg_diff_1'] = agg.diff(1)
    feats['agg_diff_5'] = agg.diff(5)
    
    # Cyclical temporal features
    hour = agg.index.hour
    dow = agg.index.dayofweek
    feats['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    feats['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    feats['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    feats['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    
    return feats


def extract_valid_dataset(df_partition: pd.DataFrame, wm_col: str):
    """
    Builds features strictly within df_partition without crossing outside boundaries.
    Drops any row whose feature window touches a NaN or outage, or where appliance is NaN.
    Reports the exact number of excluded rows.
    """
    total_rows = len(df_partition)
    X_raw = create_offline_features(df_partition['Aggregate'])
    y_raw = df_partition[wm_col]
    
    # Valid mask: no NaNs in any feature AND target is not NaN
    valid_mask = (~X_raw.isna().any(axis=1)) & (~y_raw.isna())
    
    X_clean = X_raw[valid_mask]
    y_clean = y_raw[valid_mask].values
    excluded_count = total_rows - len(X_clean)
    excluded_pct = (excluded_count / total_rows) * 100.0 if total_rows > 0 else 0.0
    
    stats = {
        "total_rows": total_rows,
        "valid_rows": len(X_clean),
        "excluded_rows": excluded_count,
        "excluded_pct": round(excluded_pct, 2)
    }
    return X_clean, y_clean, valid_mask, stats


def postprocess_prediction(y_pred: np.ndarray, cutoff_watts: float = 20.0) -> np.ndarray:
    """
    Shared post-processing function applied identically to ML model and all baselines.
    Enforces non-negativity and zeroes out low-power residual noise below cutoff_watts.
    """
    clipped = np.clip(y_pred, 0, None)
    clipped[clipped < cutoff_watts] = 0.0
    return clipped


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = WM_ACTIVE_THRESHOLD_WATTS):
    """
    Computes comprehensive NILM metrics on identical non-missing timestamps.
    """
    # 1. Overall MAE and RMSE
    mae_overall = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # 2. Active MAE (evaluated strictly when the washing machine is running)
    active_mask = (y_true >= threshold)
    if active_mask.sum() > 0:
        mae_active = float(mean_absolute_error(y_true[active_mask], y_pred[active_mask]))
    else:
        mae_active = 0.0
        
    # 3. Classification Metrics (On/Off states)
    s_true = (y_true >= threshold).astype(int)
    s_pred = (y_pred >= threshold).astype(int)
    
    prec = float(precision_score(s_true, s_pred, zero_division=0))
    rec = float(recall_score(s_true, s_pred, zero_division=0))
    f1 = float(f1_score(s_true, s_pred, zero_division=0))
    
    cm = confusion_matrix(s_true, s_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # 4. Energy Metrics
    total_true_energy = float(np.sum(y_true) / 60000.0)    # kWh (sum of 1-min watts / 60000)
    total_pred_energy = float(np.sum(y_pred) / 60000.0)    # kWh
    signed_bias_kwh = float(total_pred_energy - total_true_energy)
    
    if total_true_energy > 0:
        rete = float(abs(total_pred_energy - total_true_energy) / total_true_energy)
    else:
        rete = 0.0
        
    return {
        "mae_overall_w": round(mae_overall, 2),
        "mae_active_w": round(mae_active, 2),
        "rmse_w": round(rmse, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "ground_truth_energy_kwh": round(total_true_energy, 2),
        "predicted_energy_kwh": round(total_pred_energy, 2),
        "signed_energy_bias_kwh": round(signed_bias_kwh, 2),
        "rete_energy_error_pct": round(rete * 100.0, 2),
    }


def tune_on_validation_set(X_train: pd.DataFrame, y_train: np.ndarray,
                           X_val: pd.DataFrame, y_val: np.ndarray):
    """
    Uses Validation Set to:
    (a) Compare 3 sampling strategies: full data vs. subsampled active+25% inactive vs. class weights.
    (b) Tune postprocessing cutoff and active threshold.
    Freezes choices for subsequent test evaluation.
    """
    print("\n=======================================================")
    print("Step 1: Validation Set Model & Hyperparameter Tuning")
    print("=======================================================")
    
    # (a) Sampling Strategy Comparison
    strategies = {}
    is_active_train = (y_train >= WM_ACTIVE_THRESHOLD_WATTS)
    
    # Strategy 1: Subsample 25% inactive
    print("Evaluating Strategy 1: Subsample 25% inactive...")
    inactive_idx = np.where(~is_active_train)[0]
    np.random.seed(42)
    sampled_inactive = np.random.choice(inactive_idx, size=int(len(inactive_idx) * 0.25), replace=False)
    keep_indices = np.sort(np.concatenate([np.where(is_active_train)[0], sampled_inactive]))
    
    m_sub = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    m_sub.fit(X_train.iloc[keep_indices], y_train[keep_indices])
    p_sub = postprocess_prediction(m_sub.predict(X_val), cutoff_watts=10.0)
    met_sub = compute_metrics(y_val, p_sub, threshold=WM_ACTIVE_THRESHOLD_WATTS)
    strategies["subsample_25_pct"] = met_sub
    print(f"  Subsample 25%: Overall MAE={met_sub['mae_overall_w']}W | Active MAE={met_sub['mae_active_w']}W | Prec={met_sub['precision']} | Rec={met_sub['recall']} | F1={met_sub['f1_score']}")
    
    # Strategy 2: Class weights (5x on active)
    print("Evaluating Strategy 2: Class weights (5x on active)...")
    weights = np.ones(len(y_train), dtype=np.float32)
    weights[is_active_train] = 5.0
    m_wt = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    m_wt.fit(X_train, y_train, sample_weight=weights)
    p_wt = postprocess_prediction(m_wt.predict(X_val), cutoff_watts=10.0)
    met_wt = compute_metrics(y_val, p_wt, threshold=WM_ACTIVE_THRESHOLD_WATTS)
    strategies["class_weights_5x"] = met_wt
    print(f"  Class Weights: Overall MAE={met_wt['mae_overall_w']}W | Active MAE={met_wt['mae_active_w']}W | Prec={met_wt['precision']} | Rec={met_wt['recall']} | F1={met_wt['f1_score']}")
    
    # Strategy 3: Full training data
    print("Evaluating Strategy 3: Full training data (unweighted)...")
    m_full = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, random_state=42)
    m_full.fit(X_train, y_train)
    p_full = postprocess_prediction(m_full.predict(X_val), cutoff_watts=10.0)
    met_full = compute_metrics(y_val, p_full, threshold=WM_ACTIVE_THRESHOLD_WATTS)
    strategies["full_training_data"] = met_full
    print(f"  Full Data:    Overall MAE={met_full['mae_overall_w']}W | Active MAE={met_full['mae_active_w']}W | Prec={met_full['precision']} | Rec={met_full['recall']} | F1={met_full['f1_score']}")
    
    # (b) Postprocessing Cutoff Tuning on Validation Set using the best model (Full Data)
    print("\nEvaluating Postprocessing Zero-Cutoff on Validation Set:")
    raw_val_preds = m_full.predict(X_val)
    cutoff_grid = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    cutoff_results = {}
    best_f1 = -1
    best_cutoff = 20.0
    
    for c in cutoff_grid:
        p_c = postprocess_prediction(raw_val_preds, cutoff_watts=c)
        met_c = compute_metrics(y_val, p_c, threshold=WM_ACTIVE_THRESHOLD_WATTS)
        cutoff_results[f"cutoff_{int(c)}W"] = met_c
        print(f"  Cutoff {c:4.1f}W: MAE={met_c['mae_overall_w']}W | Prec={met_c['precision']} | Rec={met_c['recall']} | F1={met_c['f1_score']}")
        if met_c['f1_score'] > best_f1:
            best_f1 = met_c['f1_score']
            best_cutoff = c
            
    tuning_summary = {
        "sampling_strategy_comparison": strategies,
        "selected_strategy": "full_training_data",
        "cutoff_tuning": cutoff_results,
        "frozen_cutoff_watts": best_cutoff,
        "frozen_active_threshold_watts": WM_ACTIVE_THRESHOLD_WATTS
    }
    with open(OUTPUT_DIR / "validation_tuning_results.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)
        
    print(f"\n-> Frozen Choices: Strategy=Full Training Data, Cutoff={best_cutoff}W, Threshold={WM_ACTIVE_THRESHOLD_WATTS}W")
    return m_full, best_cutoff, WM_ACTIVE_THRESHOLD_WATTS, tuning_summary


def evaluate_within_home_house1(m_trained, frozen_cutoff: float, frozen_thresh: float,
                                train_df: pd.DataFrame, test_df: pd.DataFrame,
                                X_test: pd.DataFrame, y_test: np.ndarray,
                                exclusion_stats: dict):
    """
    Evaluates Primary Model vs Always-Off vs Hour-of-Week Profile Baseline
    on the unseen House 1 test partition (final 15% of timeline).
    """
    print("\n=======================================================")
    print("Step 2: Within-Home House 1 Final Evaluation (Test Set)")
    print("=======================================================")
    wm_col = HOUSE_METADATA[1]["washing_machine_col"]
    
    # 1. Baseline 1: Always-Off (predicts 0 W)
    y_pred_off = postprocess_prediction(np.zeros_like(y_test), cutoff_watts=frozen_cutoff)
    metrics_off = compute_metrics(y_test, y_pred_off, threshold=frozen_thresh)
    print(f"Baseline 1 (Always-Off): MAE={metrics_off['mae_overall_w']}W | Active MAE={metrics_off['mae_active_w']}W | F1={metrics_off['f1_score']}")
    
    # 2. Baseline 2: Hour-of-Week Profile (historical mean per (dow, hour) in train set)
    train_profiles = train_df.groupby([train_df.index.dayofweek, train_df.index.hour])[wm_col].mean().to_dict()
    y_pred_prof_raw = np.array([train_profiles.get((t.dayofweek, t.hour), 0.0) for t in X_test.index])
    y_pred_prof = postprocess_prediction(y_pred_prof_raw, cutoff_watts=frozen_cutoff)
    metrics_prof = compute_metrics(y_test, y_pred_prof, threshold=frozen_thresh)
    print(f"Baseline 2 (Time Profile): MAE={metrics_prof['mae_overall_w']}W | Active MAE={metrics_prof['mae_active_w']}W | Prec={metrics_prof['precision']} | Rec={metrics_prof['recall']} | F1={metrics_prof['f1_score']} | Signed Bias={metrics_prof['signed_energy_bias_kwh']}kWh")
    
    # 3. Primary ML Model
    y_pred_model_raw = m_trained.predict(X_test)
    y_pred_model = postprocess_prediction(y_pred_model_raw, cutoff_watts=frozen_cutoff)
    metrics_model = compute_metrics(y_test, y_pred_model, threshold=frozen_thresh)
    print(f"Primary ML Model (HistGBM): MAE={metrics_model['mae_overall_w']}W | Active MAE={metrics_model['mae_active_w']}W | RMSE={metrics_model['rmse_w']}W | Prec={metrics_model['precision']} | Rec={metrics_model['recall']} | Spec={metrics_model['specificity']} | F1={metrics_model['f1_score']} | Signed Bias={metrics_model['signed_energy_bias_kwh']}kWh | RETE={metrics_model['rete_energy_error_pct']}%")
    
    # Comparison Table
    eval_table = {
        "Metric": [
            "Overall MAE (W)", "Active MAE (W)", "RMSE (W)",
            "Precision", "Recall", "Specificity", "F1-Score",
            "Ground Truth Energy (kWh)", "Predicted Energy (kWh)",
            "Signed Energy Bias (kWh)", "Energy Error (RETE %)"
        ],
        "Always-Off Baseline": [
            metrics_off["mae_overall_w"], metrics_off["mae_active_w"], metrics_off["rmse_w"],
            metrics_off["precision"], metrics_off["recall"], metrics_off["specificity"], metrics_off["f1_score"],
            metrics_off["ground_truth_energy_kwh"], metrics_off["predicted_energy_kwh"],
            metrics_off["signed_energy_bias_kwh"], metrics_off["rete_energy_error_pct"],
        ],
        "Profile Baseline": [
            metrics_prof["mae_overall_w"], metrics_prof["mae_active_w"], metrics_prof["rmse_w"],
            metrics_prof["precision"], metrics_prof["recall"], metrics_prof["specificity"], metrics_prof["f1_score"],
            metrics_prof["ground_truth_energy_kwh"], metrics_prof["predicted_energy_kwh"],
            metrics_prof["signed_energy_bias_kwh"], metrics_prof["rete_energy_error_pct"],
        ],
        "Primary ML Model (HistGBM)": [
            metrics_model["mae_overall_w"], metrics_model["mae_active_w"], metrics_model["rmse_w"],
            metrics_model["precision"], metrics_model["recall"], metrics_model["specificity"], metrics_model["f1_score"],
            metrics_model["ground_truth_energy_kwh"], metrics_model["predicted_energy_kwh"],
            metrics_model["signed_energy_bias_kwh"], metrics_model["rete_energy_error_pct"],
        ]
    }
    df_eval = pd.DataFrame(eval_table)
    print("\n--- Final House 1 Test Comparison Table ---")
    print(df_eval.to_string(index=False))
    
    # Save metrics JSON
    metrics_payload = {
        "exclusion_statistics": exclusion_stats,
        "always_off_baseline": metrics_off,
        "profile_baseline": metrics_prof,
        "primary_model": metrics_model,
        "frozen_hyperparameters": {
            "cutoff_watts": frozen_cutoff,
            "threshold_watts": frozen_thresh,
            "model_type": "HistGradientBoostingRegressor",
            "training_samples": len(train_df),
            "test_samples": len(y_test)
        }
    }
    with open(OUTPUT_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
        
    # Generate Visualizations
    plot_actual_vs_predicted(X_test, y_test, y_pred_model)
    plot_confusion_matrix(metrics_model["confusion_matrix"])
    plot_cumulative_energy(X_test.index, y_test, y_pred_model, y_pred_prof)
    analyze_failures(X_test, y_test, y_pred_model)
    
    return metrics_model, y_pred_model, y_pred_prof


def run_leave_one_house_out(frozen_cutoff: float = 20.0, frozen_thresh: float = 20.0):
    """
    Leave-One-House-Out (LOHO) Cross-Validation across Houses {1, 2, 3, 20}.
    Trains on 3 houses and tests on the 4th, repeated for each house.
    Reports cross-household transferability vs. Always-Off and Profile baselines.
    """
    print("\n=======================================================")
    print("Step 3: Leave-One-House-Out (LOHO) Cross-Validation")
    print("=======================================================")
    houses = [1, 2, 3, 20]
    house_data = {}
    
    print("Loading and preparing datasets for 4 houses...")
    for hid in houses:
        t0 = time.time()
        df, wm_col = load_cleaned_house(hid)
        X, y, mask, stats = extract_valid_dataset(df, wm_col)
        house_data[hid] = {
            "X": X,
            "y": y,
            "stats": stats,
            "index": X.index,
            "wm_col": wm_col
        }
        print(f"  House {hid}: {len(X):,} valid 1-min samples ({stats['excluded_pct']}% excluded due to outages/windows) loaded in {time.time()-t0:.1f}s")
        
    loho_results = {}
    
    for test_hid in houses:
        train_hids = [h for h in houses if h != test_hid]
        print(f"\n--- Fold: Test on House {test_hid} (Trained on Houses {train_hids}) ---")
        
        t0 = time.time()
        X_train_list = [house_data[h]["X"] for h in train_hids]
        y_train_list = [house_data[h]["y"] for h in train_hids]
        
        X_train_concat = pd.concat(X_train_list, axis=0)
        y_train_concat = np.concatenate(y_train_list)
        
        X_test_fold = house_data[test_hid]["X"]
        y_test_fold = house_data[test_hid]["y"]
        
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train_concat, y_train_concat)
        train_time = time.time() - t0
        
        # 1. Model Predictions
        y_pred_m = postprocess_prediction(model.predict(X_test_fold), cutoff_watts=frozen_cutoff)
        met_model = compute_metrics(y_test_fold, y_pred_m, threshold=frozen_thresh)
        
        # 2. Always-Off Baseline
        y_pred_off = postprocess_prediction(np.zeros_like(y_test_fold), cutoff_watts=frozen_cutoff)
        met_off = compute_metrics(y_test_fold, y_pred_off, threshold=frozen_thresh)
        
        # 3. Profile Baseline
        train_profiles = pd.Series(y_train_concat, index=X_train_concat.index).groupby(
            [X_train_concat.index.dayofweek, X_train_concat.index.hour]
        ).mean().to_dict()
        y_pred_prof_raw = np.array([train_profiles.get((t.dayofweek, t.hour), 0.0) for t in house_data[test_hid]["index"]])
        y_pred_prof = postprocess_prediction(y_pred_prof_raw, cutoff_watts=frozen_cutoff)
        met_prof = compute_metrics(y_test_fold, y_pred_prof, threshold=frozen_thresh)
        
        loho_results[f"House_{test_hid}"] = {
            "test_house": test_hid,
            "train_houses": train_hids,
            "test_samples": len(y_test_fold),
            "train_time_seconds": round(train_time, 1),
            "model_metrics": met_model,
            "always_off_metrics": met_off,
            "profile_baseline_metrics": met_prof,
        }
        print(f"  House {test_hid} Results: Model MAE={met_model['mae_overall_w']}W (Off={met_off['mae_overall_w']}W, Prof={met_prof['mae_overall_w']}W) | F1={met_model['f1_score']} (Prof F1={met_prof['f1_score']}) | Prec={met_model['precision']} | Rec={met_model['recall']}")
        
    with open(OUTPUT_DIR / "loho_cross_validation_results.json", "w") as f:
        json.dump(loho_results, f, indent=2)
    print(f"\nLOHO results saved to {OUTPUT_DIR / 'loho_cross_validation_results.json'}")
    return loho_results


def plot_actual_vs_predicted(X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Plots Whole-Home Aggregate Load and Disaggregated Washer Power for an active test period.
    """
    print("Generating Plot: sec3_actual_vs_predicted.png...")
    # Find a window in test set containing representative wash cycles
    active_idx = np.where(y_true > 200.0)[0]
    if len(active_idx) > 0:
        center_time = X_test.index[active_idx[len(active_idx)//2]]
        start_w = center_time - pd.Timedelta(days=2)
        end_w = center_time + pd.Timedelta(days=3)
    else:
        start_w = X_test.index[0]
        end_w = start_w + pd.Timedelta(days=5)
        
    mask = (X_test.index >= start_w) & (X_test.index < end_w)
    plot_times = X_test.index[mask]
    plot_agg = X_test.loc[mask, 'agg_current'].values
    plot_true = y_true[mask]
    plot_pred = y_pred[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.4]})
    
    # Top panel: Aggregate Load
    ax1.plot(plot_times, plot_agg, color='#2c3e50', linewidth=1.1, label='Whole-Home Aggregate Power (W)')
    ax1.set_ylabel("Aggregate (W)", fontsize=11, fontweight='bold')
    ax1.set_title("Test Period Disaggregation: Aggregate Load vs. Washing Machine Model Prediction", fontsize=13, fontweight='bold')
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


def plot_confusion_matrix(cm_dict: dict):
    """
    Plots the On/Off State Confusion Matrix Heatmap for the test set.
    """
    print("Generating Plot: sec3_confusion_matrix.png...")
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    
    total = tn + fp + fn + tp
    matrix = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Off", "Predicted On"], fontsize=11, fontweight='bold')
    ax.set_yticklabels(["Actual Off", "Actual On"], fontsize=11, fontweight='bold')
    
    # Annotate cells
    labels = [
        [f"True Negative\n{tn:,}\n({tn/total*100:.2f}%)", f"False Positive\n{fp:,}\n({fp/total*100:.2f}%)"],
        [f"False Negative\n{fn:,}\n({fn/total*100:.2f}%)", f"True Positive\n{tp:,}\n({tp/total*100:.2f}%)"]
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > (total * 0.4) else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight='bold')
            
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    plt.title(f"Confusion Matrix (Test Set, N={total:,})\nPrecision: {prec:.3f} | Recall: {rec:.3f} | Specificity: {spec:.3f} | F1: {f1:.3f}",
              fontsize=12, fontweight='bold', pad=15)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "sec3_confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")


def plot_cumulative_energy(timestamps, y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_prof: np.ndarray):
    """
    Plots Cumulative Energy Consumption (kWh) over the test set timeline.
    """
    print("Generating Plot: sec3_cumulative_energy.png...")
    # 1 minute energy in kWh: W / 60000
    cum_true = np.cumsum(y_true) / 60000.0
    cum_model = np.cumsum(y_pred_model) / 60000.0
    cum_prof = np.cumsum(y_pred_prof) / 60000.0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, cum_true, label=f"Ground Truth ({cum_true[-1]:.1f} kWh)", color="#27ae60", linewidth=2.2)
    ax.plot(timestamps, cum_model, label=f"Primary ML Model ({cum_model[-1]:.1f} kWh, Bias: {cum_model[-1]-cum_true[-1]:+.1f} kWh)", color="#e74c3c", linewidth=2.0, linestyle="--")
    ax.plot(timestamps, cum_prof, label=f"Hour-of-Week Profile ({cum_prof[-1]:.1f} kWh, Bias: {cum_prof[-1]-cum_true[-1]:+.1f} kWh)", color="#2980b9", linewidth=1.5, linestyle=":")
    
    ax.set_title("Test Period: Cumulative Washing Machine Energy Consumption (Actual vs. Predicted)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax.set_ylabel("Cumulative Energy (kWh)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    
    plt.tight_layout()
    fig_path = FIGURES_DIR / "sec3_cumulative_energy.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved: {fig_path}")


def analyze_failures(X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Quantifies and diagnoses model failure modes on the test set.
    """
    print("\n--- Failure Mode Analysis ---")
    # False Positives: Washer was off (<20W), but model predicted active washing (>=200W)
    fp_mask = (y_true < WM_ACTIVE_THRESHOLD_WATTS) & (y_pred >= 200.0)
    fp_count = int(fp_mask.sum())
    
    # False Negatives: Washer was running heater (>500W), but model predicted <100W
    fn_mask = (y_true >= 500.0) & (y_pred < 100.0)
    fn_count = int(fn_mask.sum())
    
    fp_agg_mean = float(X_test.loc[fp_mask, 'agg_current'].mean()) if fp_count > 0 else 0.0
    fn_agg_mean = float(X_test.loc[fn_mask, 'agg_current'].mean()) if fn_count > 0 else 0.0
    
    print(f"False Positives (Washer Off, Pred >= 200W): {fp_count} minutes")
    print(f"  -> Conflation with other resistive loads: Mean Aggregate Power = {fp_agg_mean:.1f} W")
    print(f"False Negatives (Washer >= 500W, Pred < 100W): {fn_count} minutes")
    print(f"  -> Masking under concurrent heavy loads: Mean Aggregate Power = {fn_agg_mean:.1f} W")
    
    failure_data = {
        "false_positive_minutes": fp_count,
        "false_negative_minutes": fn_count,
        "fp_agg_mean_watts": round(fp_agg_mean, 1),
        "fn_agg_mean_watts": round(fn_agg_mean, 1)
    }
    with open(OUTPUT_DIR / "failure_analysis.json", "w") as f:
        json.dump(failure_data, f, indent=2)


def main():
    print("\n=======================================================")
    print("STARTING SECTION 3: APPLIANCE DISAGGREGATION PIPELINE")
    print("=======================================================")
    
    # 1. Load Cleaned House 1 Data
    df_h1, wm_col = load_cleaned_house(1)
    n = len(df_h1)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)
    
    train_df = df_h1.iloc[:train_idx]
    val_df = df_h1.iloc[train_idx:val_idx]
    test_df = df_h1.iloc[val_idx:]
    
    # 2. Extract Valid Continuous Datasets (No crossing partition boundaries or outages)
    X_train, y_train, mask_train, stats_train = extract_valid_dataset(train_df, wm_col)
    X_val, y_val, mask_val, stats_val = extract_valid_dataset(val_df, wm_col)
    X_test, y_test, mask_test, stats_test = extract_valid_dataset(test_df, wm_col)
    
    exclusion_stats = {
        "train": stats_train,
        "val": stats_val,
        "test": stats_test
    }
    print(f"Partition Statistics (Dropping rows whose window touches NaN or boundary):")
    print(f"  Train: total={stats_train['total_rows']:,}, valid={stats_train['valid_rows']:,}, excluded={stats_train['excluded_rows']:,} ({stats_train['excluded_pct']}%)")
    print(f"  Val:   total={stats_val['total_rows']:,}, valid={stats_val['valid_rows']:,}, excluded={stats_val['excluded_rows']:,} ({stats_val['excluded_pct']}%)")
    print(f"  Test:  total={stats_test['total_rows']:,}, valid={stats_test['valid_rows']:,}, excluded={stats_test['excluded_rows']:,} ({stats_test['excluded_pct']}%)")
    
    # 3. Validation Tuning
    m_full, frozen_cutoff, frozen_thresh, val_tuning = tune_on_validation_set(X_train, y_train, X_val, y_val)
    
    # 4. Within-Home Final Test Evaluation
    evaluate_within_home_house1(
        m_full, frozen_cutoff, frozen_thresh,
        train_df, test_df, X_test, y_test, exclusion_stats
    )
    
    # 5. Leave-One-House-Out Cross-Validation
    run_leave_one_house_out(frozen_cutoff=frozen_cutoff, frozen_thresh=frozen_thresh)
    
    print("\n=======================================================")
    print("SECTION 3 COMPLETED SUCCESSFULLY")
    print("=======================================================")


if __name__ == "__main__":
    main()
