# RESULTS.md: Washing Machine EDA and Appliance Disaggregation Evaluation

## 1. Executive Summary
This report presents the findings from the exploratory data analysis (EDA) of washing machine usage across four REFIT households (Houses 1, 2, 3, and 20) and evaluates the performance of a supervised machine learning disaggregation model (HistGradientBoosting) against two baseline models using strictly non-leaking chronological validation and cross-household Leave-One-House-Out (LOHO) cross-validation.

---

## 2. Washing Machine Exploratory Data Analysis (EDA)

### 2.1. Dataset Selection & Metadata
Four households with dedicated, verified washing machine sub-metering channels were analyzed at 1-minute resolution using the official cleaned REFIT dataset via the shared data loader (`load_cleaned_house`):
* **House 1**: Channel 5 (`Appliance5`), Beko WMC6140, Detached (4-bed), 2 occupants.
* **House 2**: Channel 2 (`Appliance2`), LG F1289TD, Semi-detached (3-bed), 4 occupants.
* **House 3**: Channel 6 (`Appliance6`), Model Unknown, Detached (3-bed), 2 occupants.
* **House 20**: Channel 4 (`Appliance4`), Model Unknown, Detached (3-bed), 2 occupants.

Per dataset documentation (`CLEAN_READ_ME_081116.txt`), corrupted intervals flagged with `Issues == 1` were discarded prior to analysis.

---

### 2.2. Active Cycle Detection Rule
Washing machines are multi-stage cyclic appliances with distinct physical phases:
1. **Water fill & pulsed heating**: Resistive immersion element bursts (~1,800–2,400 W) lasting 3–5 minutes each.
2. **Wash agitation / tumbling**: Periodic motor pulses (~100–300 W).
3. **Soak / resting pauses**: Power drops to idle/standby (< 20 W) for several minutes.
4. **Drain & spin extraction**: Drum acceleration pulses (~400–800 W).

#### Rule Definition (Plain Language):
> "An active washing machine cycle is defined as a period where power exceeds a standby threshold of **20 Watts** for a cumulative operating duration of at least **15 minutes**. To ensure that standard multi-stage wash sequences are not artificially fragmented by internal soaking or resting lulls, any sub-threshold dips lasting **up to 8 consecutive minutes** are bridged and treated as part of the same continuous cycle. Furthermore, to eliminate non-wash electrical transients or sensor noise, the candidate cycle must reach a peak power of at least **600 Watts** or accumulate at least **0.05 kWh** of energy."

---

### 2.3. Availability & Data Quality Summary
All metrics are derived directly from `output/washing_machine_eda_summary.json`:

| Household | Occupancy & Type | Date Range | Duration (Days) | Total Minutes | Missing / Outage (%) | Off / Standby (%) | Total Detected Cycles | Cycles / Week | Mean Duration (min) | Median Duration (min) | Mean Energy (kWh) | Mean Peak Power (W) | Total Energy (kWh) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **House 1** | 2 occ, Detached | 2013-10-09 to 2015-07-10 | 639.0 | 920,091 | 12.91% | **98.21%** | **375** | **4.11** | 41.5 | 33.0 | 0.332 | 2,162.3 | 124.6 |
| **House 2** | 4 occ, Semi-det | 2013-09-17 to 2015-05-28 | 617.4 | 889,078 | 24.65% | **95.38%** | **336** | **3.81** | 98.0 | 96.0 | 0.590 | 2,038.0 | 198.3 |
| **House 3** | 2 occ, Detached | 2013-09-25 to 2015-06-02 | 614.6 | 885,095 | 14.98% | **94.56%** | **627** | **7.14** | 64.3 | 65.0 | 0.500 | 1,774.6 | 313.2 |
| **House 20** | 2 occ, Detached | 2014-03-20 to 2015-06-23 | 460.3 | 662,864 | 6.01% | **97.59%** | **198** | **3.01** | 78.4 | 74.0 | 0.401 | 2,211.2 | 79.3 |

#### Cycle Detection Sensitivity Analysis
To assess the robustness of our finite-state machine, we evaluated cycle counts across a grid of power thresholds (10 W, 20 W, 50 W) and bridging gaps (5 min, 8 min, 12 min):

| Configuration | House 1 (Ch 5) | House 2 (Ch 2) | House 3 (Ch 6) | House 20 (Ch 4) |
| :--- | :---: | :---: | :---: | :---: |
| **Threshold 10 W, Gap 5 min** | 383 | 337 | 636 | 201 |
| **Threshold 10 W, Gap 8 min** | 376 | 332 | 627 | 191 |
| **Threshold 10 W, Gap 12 min** | 363 | 321 | 618 | 186 |
| **Threshold 20 W, Gap 5 min** | 386 | 335 | 637 | 203 |
| **Threshold 20 W, Gap 8 min (Selected)** | **375** | **336** | **627** | **198** |
| **Threshold 20 W, Gap 12 min** | 363 | 327 | 618 | 188 |
| **Threshold 50 W, Gap 5 min** | 385 | 364 | 639 | 202 |
| **Threshold 50 W, Gap 8 min** | 371 | 343 | 631 | 202 |
| **Threshold 50 W, Gap 12 min** | 361 | 332 | 621 | 189 |

*Finding*: Cycle counts are stable across the grid. Extending the bridging gap from 5 to 12 minutes consolidates multi-stage programs by ~5–7%, while moving threshold from 10 W to 50 W has minimal impact because washing machine standby power is typically below 3 W.

---

### 2.4. Plot Interpretations

#### Plot (i): Representative 24-Hour Electricity Load Trace (`figures/sec2_24h_trace.png`)
* **Visual Content**: Compares whole-home aggregate electricity load against the sub-metered washing machine load over a full 24-hour period on **2015-02-23** in House 1. (Note: previous drafts mistakenly plotted 2013-11-17, a day when the washing machine remained idle at 0.0 W all day).
* **Actual Observations**:
  * Baseline household load hovers between 150 W and 400 W, with periodic ~100 W refrigerator compressor cycles.
  * During the active laundry event (active for 106 minutes, total energy 0.35 kWh), the washing machine peaks at **2,367.8 W** as the water heating element engages.
  * Other household surges occur earlier and later in the day (reaching 1,500–2,000 W) completely independent of the washing machine, showing why raw magnitude alone is insufficient for disaggregation.

#### Plot (ii): Anatomy of a Detected Wash Cycle (`figures/sec2_detected_cycles.png`)
* **Visual Content**: A zoomed trace of a genuine wash cycle in House 1 showing detected start and end boundary markers and stage annotations without legend overlap.
* **Actual Observations**:
  * **Water Heating Dynamics**: The heating element does not draw power as a single uninterrupted 12-minute block. Instead, it fires in **distinct ~3–5 minute pulsed bursts** (~2,200–2,400 W) interspersed with initial drum tumbling to regulate water temperature.
  * **Wash & Agitation**: Tumbling produces low-magnitude motor pulses (~150–300 W) separated by brief resting pauses. The 8-minute gap bridge successfully prevents premature cycle termination.
  * **Rinse & Spin Extraction**: High-speed drum spinning produces variable motor peaks (~400–600 W) followed by clean return to 0 W.

#### Plot (iii): Cross-Household Cycle Characteristic Comparison (`figures/sec2_cross_house_comparison.png`)
* **Visual Content**: Three side-by-side boxplots comparing Energy per Cycle (kWh), Cycle Duration (min), and Peak Power (W) across Houses 1, 2, 3, and 20.
* **Actual Observations**:
  * **Cycle Duration**: House 1 features short, rapid cycles (median 33.0 min; mean 41.5 min), whereas House 2 has long cycles (median 96.0 min; mean 98.0 min). House 3 averages 64.3 min and House 20 averages 78.4 min.
  * **Energy Consumption**: House 2 consumes the highest average energy per cycle (mean 0.590 kWh, median 0.602 kWh). House 1 is the most energy-efficient (mean 0.332 kWh, median 0.294 kWh).
  * **Peak Power**: All four homes exhibit heating peaks centered between 1,750 W and 2,250 W, reflecting standard UK 230V resistive immersion heating elements.

#### Plot (iv): Temporal Usage Patterns Across 4 REFIT Homes (`figures/sec2_temporal_patterns.png`)
* **Visual Content**: Aggregate distributions of cycle start times by hour of day (0–23) and day of week across all 1,536 detected wash cycles.
* **Actual Observations (Directly from Data)**:
  * **Weekday Distribution**:
    * **Monday is the highest laundry day**: 298 cycles (**19.40%**).
    * Followed by Tuesday: 267 cycles (**17.38%**), Wednesday: 243 cycles (**15.82%**), Friday: 228 cycles (**14.84%**), and Thursday: 213 cycles (**13.87%**).
    * **Weekends are the lowest**: Sunday had 146 cycles (**9.51%**) and Saturday had 141 cycles (**9.18%**). The previous draft's claim of "weekend peaks" was directly contradicted by the empirical data.
  * **Diurnal Pattern**:
    * Morning concentration: **44.92%** of all cycles (690 cycles) start between **08:00 and 12:00**.
    * Secondary afternoon peak occurs around 13:00–15:00.
    * **Overnight Laundry in House 1**: Exactly **88 cycles** (**23.47%** of House 1 cycles) started between **00:00 and 06:00**, reflecting overnight Economy 7 off-peak tariff scheduling.

---

## 3. Disaggregation Model Formulation & Evaluation

### 3.1. Offline Problem Formulation & Feature Architecture
The disaggregation model operates strictly in an **OFFLINE retrospective setting** (designed for utility end-of-month billing decomposition, carbon footprint audits, and retrospective load profiling). It uses future lookahead and centered windows:
$$\hat{y}(t) = f\big(\mathbf{X}_{\text{agg}}(t)\big)$$
where $\mathbf{X}_{\text{agg}}(t)$ is a 31-minute receptive field:
* Instantaneous aggregate load: $x(t)$
* Past historical lags: $x(t-1), x(t-2), x(t-3), x(t-5), x(t-10), x(t-15)$
* Future lead context: $x(t+1), x(t+2), x(t+3), x(t+5), x(t+10), x(t+15)$
* Centered rolling statistics over windows of 5, 15, 30, and 60 minutes (mean, min, max, standard deviation)
* Rate-of-change edge features: $\Delta_1 x(t) = x(t) - x(t-1)$, $\Delta_5 x(t) = x(t) - x(t-5)$
* Cyclical temporal encodings: $\sin / \cos$ of hour-of-day and day-of-week.

> **Offline Disclosure**: Because this model incorporates future aggregate power up to $t+15$ minutes and centered 60-minute rolling statistics, it is strictly an **offline** model and cannot be deployed in online real-time stream processing without introducing latency.

---

### 3.2. Temporal Continuity & Window Integrity
Features were engineered strictly within continuous valid segments on the 1-minute grid. Rows whose feature window touched an outage (> 3 min NaN) or partition boundary were excluded:
* **Train Set (First 70% of timeline)**: 644,063 total rows $\rightarrow$ **538,544 valid rows** (105,519 excluded, 16.38%).
* **Validation Set (Next 15% of timeline)**: 138,014 total rows $\rightarrow$ **124,765 valid rows** (13,249 excluded, 9.60%).
* **Test Set (Final 15% of timeline)**: 138,014 total rows $\rightarrow$ **107,005 valid rows** (31,009 excluded, 22.47%).

---

### 3.3. Validation Set Tuning & Frozen Decisions
Before touching the test set, the Validation set was used to resolve two key architectural decisions:

#### 1. Training Sampling Strategy Comparison (Validation Set):
| Strategy | Overall MAE (W) | Active MAE (W) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1: Subsample 25% Inactive** | 16.56 W | 442.86 W | 0.1886 | **0.9037** | 0.3121 |
| **Strategy 2: Class Weights (5x on Active)**| 16.29 W | 456.01 W | 0.1812 | 0.8964 | 0.3015 |
| **Strategy 3: Full Training Data (Unweighted)**| **13.06 W** | 500.86 W | **0.3206** | 0.7241 | **0.4444** |

*Decision*: Training on full unweighted data yielded substantially fewer false positives, boosting Precision from 0.1886 to 0.3206 and F1-score from 0.3121 to 0.4444.

#### 2. Postprocessing Zero-Cutoff Tuning (Validation Set):
| Zero-Cutoff | Overall MAE (W) | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 W | 13.57 W | 0.3206 | 0.7241 | 0.4444 |
| 10.0 W | 13.06 W | 0.3206 | 0.7241 | 0.4444 |
| 20.0 W | 12.80 W | 0.3206 | 0.7241 | 0.4444 |
| **30.0 W (Selected)** | **12.65 W** | **0.3671** | 0.6356 | **0.4654** |

*Decision*: A post-processing cutoff of **30.0 W** was frozen alongside the physical active threshold of **20.0 W**. This shared post-processing function was applied identically to the ML model and both baselines.

---

### 3.4. Final Within-Home Test Evaluation (House 1: 107,005 Minutes)
Evaluated on the exact same unseen test timestamps:

| Metric | Always-Off Baseline | Hour-of-Week Profile Baseline | Primary ML Model (HistGBM) |
| :--- | :---: | :---: | :---: |
| **Overall MAE (W)** | 10.40 W | 12.43 W | **5.91 W** |
| **Active MAE (W)** | 512.91 W | 509.75 W | **200.54 W** |
| **RMSE (W)** | 132.10 W | 132.12 W | **66.35 W** |
| **Precision** | 0.0000 | 0.0298 | **0.5453** |
| **Recall** | 0.0000 | 0.0828 | **0.7512** |
| **Specificity** | **1.0000** | 0.9443 | **0.9871** |
| **F1-Score** | 0.0000 | 0.0438 | **0.6319** |
| **Ground Truth Energy (kWh)** | 18.54 kWh | 18.54 kWh | 18.54 kWh |
| **Predicted Energy (kWh)** | 0.00 kWh | 3.86 kWh | **15.32 kWh** |
| **Signed Energy Bias (kWh)** | -18.54 kWh | -14.68 kWh | **-3.22 kWh** |
| **Energy Error (RETE %)** | 100.00% | 79.16% | **17.35%** |

#### Test Set Confusion Matrix (On/Off Classification):
* **True Negatives (TN)**: 103,489 minutes (**96.71%**)
* **False Positives (FP)**: 1,354 minutes (**1.27%**)
* **False Negatives (FN)**: 538 minutes (**0.50%**)
* **True Positives (TP)**: 1,624 minutes (**1.52%**)

> **Critical Note on RETE vs. Accuracy**: The Relative Error in Total Energy (RETE) is **17.35%** (predicted 15.32 kWh vs. actual 18.54 kWh). It is critical to state that **RETE is an aggregate energy error metric, NOT billing accuracy**. Because overpredictions (FP) and underpredictions (FN) partially cancel out over long integration windows, RETE can make a model look more accurate than its point-by-point precision (54.53%) implies.

---

### 3.5. Cross-Household Generalization: Leave-One-House-Out (LOHO) Cross-Validation
To evaluate how well the model generalizes across different homes, appliances, and user behaviors, we performed 4-fold Leave-One-House-Out cross-validation across all four houses:

| Test Fold | Trained On | Test Samples (1-min) | Model Overall MAE (W) | Baseline Always-Off MAE (W) | Baseline Profile MAE (W) | Model F1-Score | Baseline Profile F1-Score | Model Precision | Model Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **House 1** | Houses 2, 3, 20 | 770,373 | 11.72 W | 9.46 W | 18.75 W | **0.3040** | 0.0677 | 0.2149 | 0.5194 |
| **House 2** | Houses 1, 3, 20 | 652,422 | 24.08 W | 17.69 W | 23.31 W | **0.2417** | 0.1330 | 0.2127 | 0.2798 |
| **House 3** | Houses 1, 2, 20 | 696,834 | 44.40 W | 24.59 W | 27.14 W | **0.3615** | 0.1984 | 0.2819 | 0.5037 |
| **House 20** | Houses 1, 2, 3 | 618,239 | 9.68 W | 7.48 W | 16.42 W | **0.2307** | 0.1354 | 0.1972 | 0.2779 |

#### Takeaways from Cross-Household Evaluation:
1. **Model Beats Time-of-Week Baselines Across All Houses**: On every test house, the ML model substantially outperforms the Hour-of-Week Profile baseline on F1-score (e.g. House 1: 0.3040 vs 0.0677; House 3: 0.3615 vs 0.1984).
2. **Transfer Degradation**: Compared to the within-home split (F1 = 0.6319), cross-household F1 drops to 0.23–0.36. This performance drop highlights appliance heterogeneity: different drum sizes, motor efficiencies, wash durations (House 1 ~41 min vs House 2 ~98 min), and unmetered background appliances make universal NILM transfer challenging without transfer calibration.
3. **Always-Off Paradox**: In all four homes, the Always-Off baseline achieves a lower overall MAE than both the Profile baseline and the ML model simply because washing machines operate only 1.8–5.4% of the time. Predicting zero everywhere minimizes aggregate MAE while providing zero utility value.

---

### 3.6. Failure Mode Analysis (Test Set)

#### Failure Mode 1: False Positives (Conflation with Resistive Loads)
* **Observation**: In the test set, **245 minutes** produced false positive predictions $\ge 200\text{ W}$ when the washing machine was completely off.
* **Root Cause**: During these false positive intervals, the average household aggregate power was **2,802.6 Watts**. Other high-power resistive loads—specifically electric kettles (~2.5–3 kW), dishwashers on wash/dry cycles, or electric ovens—draw active power profiles that mimic the heating phase of a washing machine.

#### Failure Mode 2: False Negatives (Masking Under Heavy Concurrent Load)
* **Observation**: Exactly **24 minutes** suffered severe underprediction ($y_{\text{true}} \ge 500\text{ W}$ while $\hat{y} < 100\text{ W}$).
* **Root Cause**: These misses occurred when the household aggregate load was extraordinarily high and erratic (mean aggregate **3,085.8 Watts**), masking the subtle tumbling or ramp-down stages of the washer inside massive background noise.
