# The Complete REFIT NILM & Data Engineering Encyclopedia
## Exhaustive Technical Reference, Architecture, File-by-File Code Walkthrough, and Methodology Guide

**Project:** Non-Intrusive Load Monitoring (NILM) for Washing Machine Disaggregation  
**Dataset:** REFIT Electrical Load Measurements (Longitudinal UK Dataset, EPSRC Grants EP/K002708/1 and EP/K002759/1)  
**Target Enterprise:** Flock Energy (Utility Operations, Product Strategy, and Demand-Side Management)  
**Author:** Candidate Take-Home Engineering & Machine Learning Defense  

---

# TABLE OF CONTENTS
1. [Executive Overview & Problem Statement](#1-executive-overview--problem-statement)
2. [Complete Repository Directory & File Inventory](#2-complete-repository-directory--file-inventory)
3. [Deep-Dive: Configuration & Constants (`src/config.py`)](#3-deep-dive-configuration--constants-srcconfigpy)
4. [Deep-Dive: Shared Data Utilities (`src/utils.py`)](#4-deep-dive-shared-data-utilities-srcutilspy)
5. [Deep-Dive: Unit Testing Suite (`tests/test_gaps.py`)](#5-deep-dive-unit-testing-suite-teststest_gapspy)
6. [Deep-Dive: Section 1 Raw Telemetry Inspection (`src/data_inspection.py`)](#6-deep-dive-section-1-raw-telemetry-inspection-srcdata_inspectionpy)
7. [Deep-Dive: Section 1 Raw Cleaning & Resampling Engine (`src/data_cleaning.py`)](#7-deep-dive-section-1-raw-cleaning--resampling-engine-srcdata_cleaningpy)
8. [Deep-Dive: Section 2 Washing Machine EDA Engine (`src/eda_washing_machine.py`)](#8-deep-dive-section-2-washing-machine-eda-engine-srceda_washing_machinepy)
9. [Deep-Dive: Cycle Detection State Machine & Sensitivity Matrix](#9-deep-dive-cycle-detection-state-machine--sensitivity-matrix)
10. [Deep-Dive: Multi-Household Empirical Findings (1,536 Cycles)](#10-deep-dive-multi-household-empirical-findings-1536-cycles)
11. [Deep-Dive: Section 3 Offline Feature Engineering (`src/disaggregation_model.py`)](#11-deep-dive-section-3-offline-feature-engineering-srcdisaggregation_modelpy)
12. [Deep-Dive: Zero-Leakage Chronological Partitioning & Row Exclusion](#12-deep-dive-zero-leakage-chronological-partitioning--row-exclusion)
13. [Deep-Dive: Validation Set Tuning & The Subsampling Paradox](#13-deep-dive-validation-set-tuning--the-subsampling-paradox)
14. [Deep-Dive: Model Architecture & HistGradientBoosting Selection](#14-deep-dive-model-architecture--histgradientboosting-selection)
15. [Deep-Dive: Final Within-Home Test Evaluation & Confusion Matrix](#15-deep-dive-final-within-home-test-evaluation--confusion-matrix)
16. [Deep-Dive: Leave-One-House-Out (LOHO) 4-Fold Cross-Validation](#16-deep-dive-leave-one-house-out-loho-4-fold-cross-validation)
17. [Deep-Dive: Quantitative Failure Mode Analysis](#17-deep-dive-quantitative-failure-mode-analysis)
18. [Visualizations Gallery & Detailed Interpretation (7 Figures)](#18-visualizations-gallery--detailed-interpretation-7-figures)
19. [Deep-Dive: Section 4 Utility Strategy & M&V Design (`Recommendation.md`)](#19-deep-dive-section-4-utility-strategy--mv-design-recommendationmd)
20. [Deep-Dive: Geographic Adaptation (UK vs. India) & 30-Min Meters](#20-deep-dive-geographic-adaptation-uk-vs-india--30-min-meters)
21. [Pipeline Orchestrator & Interactive Notebook Walkthrough](#21-pipeline-orchestrator--interactive-notebook-walkthrough)
22. [AI Usage Transparency, Verification Protocol & Academic Citations](#22-ai-usage-transparency-verification-protocol--academic-citations)

---

# 1. Executive Overview & Problem Statement

### 1.1. The Energy Sector Dilemma
In modern electrical utility operations, residential consumers account for a substantial fraction of peak grid demand. High-power domestic appliances—specifically wet appliances (washing machines, washer-dryers, dishwashers) with internal water heaters (~2.0–2.5 kW)—exert severe, localized thermal stress on low-voltage distribution transformers.

Utilities seek to offer personalized energy breakdowns ("Where Your Energy Went"), design dynamic Time-of-Use (ToU) tariffs, and incentivize peak-shaving demand response. However, installing hardware sub-meters (Individual Appliance Monitors, or IAMs) on individual appliances across hundreds of thousands of customer homes is economically unviable (exceeding hundreds of pounds per household in hardware and installation costs).

### 1.2. The Solution: Non-Intrusive Load Monitoring (NILM)
Non-Intrusive Load Monitoring (NILM) solves this challenge algorithmically: given only the single-point, whole-home aggregate power measurement $P_{\text{agg}}(t)$ captured by a standard utility smart meter, machine learning models mathematically decompose (disaggregate) the composite signal to infer the operating states and power consumption of specific sub-appliances:

$$\hat{y}_i(t) = f\big(P_{\text{agg}}(t-k), \dots, P_{\text{agg}}(t), \dots, P_{\text{agg}}(t+m)\big)$$

### 1.3. Scope & Explicit Operational Boundaries
* **Target Appliance**: Domestic Washing Machine (multi-stage electromechanical load).
* **Operational Mode**: **Strictly OFFLINE Retrospective Disaggregation**. Designed for end-of-month billing decomposition, carbon accounting, retrospective tariff audits, and aggregate demand forecasting.
* **Explicit Disclaimers & Boundaries**:
  * *Not a Real-Time Circuit Breaker*: The model incorporates a 15-minute future lookahead ($t+15$) and cannot run in zero-latency streaming control loops.
  * *Not for Legal Sub-Billing*: Inherent estimation variance (active MAE ~200 W, precision ~55%) violates statutory legal sub-metering accuracy requirements.
  * *Aggregated DSM Feasibility*: Highly effective for population-level load shifting and targeted consumer nudging.

---

# 2. Complete Repository Directory & File Inventory

Every file in the repository serves a distinct, modular purpose in the data engineering and machine learning lifecycle:

```
refit_disaggregation/
├── data/
│   ├── raw/                                    # Raw multi-gigabyte REFIT telemetry
│   │   ├── RAW_House1_Part1.csv                # 4,377,877 rows (0 = missing sub-meters)
│   │   └── RAW_House1_Part2.csv                # 4,140,886 rows (NaN = missing sub-meters)
│   └── cleaned/                                # Official Cleaned REFIT CSV files
│       ├── CLEAN_House1.csv                    # House 1 (Washing machine: Appliance5)
│       ├── CLEAN_House2.csv                    # House 2 (Washing machine: Appliance2)
│       ├── CLEAN_House3.csv                    # House 3 (Washing machine: Appliance6)
│       └── CLEAN_House20.csv                   # House 20 (Washing machine: Appliance4)
├── src/                                        # Core modular Python package
│   ├── __init__.py                             # Package initialization
│   ├── config.py                               # Central constants, paths, thresholds, metadata
│   ├── utils.py                                # Shared gap engine, outage masks, shared data loader
│   ├── data_inspection.py                      # Section 1: Raw data quality auditing & anomalies
│   ├── data_cleaning.py                        # Section 1: Merge, deduplicate, resample, gap fill
│   ├── eda_washing_machine.py                  # Section 2: 4-house cycle detection & pattern analysis
│   └── disaggregation_model.py                 # Section 3: Feature engine, validation tuning, HistGBM, LOHO
├── tests/                                      # Automated testing suite
│   └── test_gaps.py                            # Pytest verifying short vs long gap handling
├── output/                                     # Generated artifacts (never hand-edited)
│   ├── House1_cleaned_1min.parquet             # Resampled uniform 1-minute Parquet file (10.9 MB)
│   ├── raw_inspection_summary.json             # Section 1 inspection audit numbers
│   ├── washing_machine_eda_summary.json        # Section 2 4-house cycle metrics & sensitivity
│   ├── validation_tuning_results.json          # Section 3 sampling & cutoff grid search
│   ├── model_evaluation_metrics.json           # Section 3 within-home House 1 test metrics
│   ├── loho_cross_validation_results.json      # Section 3 4-fold cross-household metrics
│   ├── failure_analysis.json                   # Section 3 false positive/negative diagnoses
│   └── figures/                                # Publication-grade visualizations
│       ├── sec1_raw_vs_cleaned.png             # Section 1: 3-panel raw vs cleaned trace
│       ├── sec2_24h_trace.png                  # Section 2: 24-hour trace on active day 2015-02-23
│       ├── sec2_detected_cycles.png            # Section 2: Zoomed wash cycle with pulsed heating
│       ├── sec2_cross_house_comparison.png     # Section 2: Boxplots of duration, energy, peak
│       ├── sec2_temporal_patterns.png          # Section 2: Diurnal & day-of-week usage distributions
│       ├── sec3_actual_vs_predicted.png        # Section 3: Test period whole-home vs predicted trace
│       ├── sec3_confusion_matrix.png           # Section 3: On/Off state confusion matrix heatmap
│       └── sec3_cumulative_energy.png          # Section 3: Cumulative energy accounting trace (kWh)
├── preparation/                                # Technical interview preparation package
│   ├── COMPLETE_PROJECT_ENCYCLOPEDIA.md        # This exhaustive master reference document
│   ├── overview.md                             # Deep technical analysis across structured sections
│   ├── Intro.md                                # Spoken first-person 3-minute technical pitch
│   └── Q&A.md                                  # Exhaustive 12-category technical defense guide
├── notebooks/                                  # Interactive walkthrough
│   └── refit_end_to_end_pipeline.ipynb         # Jupyter notebook running the full pipeline
├── run_pipeline.py                             # Master CLI pipeline execution script
├── requirements.txt                            # Pinned Python package dependencies
├── DATA.md                                     # Raw inspection and cleaning report
├── RESULTS.md                                  # Formal EDA and modeling evaluation report
├── Recommendation.md                           # 1-page utility executive summary for Flock Energy
└── README.md                                   # Project landing page, setup, citations, and AI disclosure
```

---

# 3. Deep-Dive: Configuration & Constants (`src/config.py`)

### 3.1. Purpose & Responsibilities
`src/config.py` acts as the single source of truth for all filesystem paths, physical power thresholds, cycle detection parameters, and household appliance channel mappings. It prevents hardcoded numbers from leaking into algorithmic code.

### 3.2. Code Structure & Key Constants
```python
# Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Physical Power Limits (Domain Constraints)
POWER_UPPER_LIMIT_AGG = 15000.0  # Watts (UK residential single-phase limit ~15-23 kW)
POWER_UPPER_LIMIT_IAM = 4000.0   # Watts (Standard UK 13A wall socket ~3.12 kW)

# Resampling and Gap Parameters
RESAMPLE_INTERVAL = '1min'       # Target uniform grid
MAX_INTERPOLATION_MINUTES = 3    # Maximum allowable gap for linear interpolation

# Washing Machine Cycle Detection Parameters
WM_ACTIVE_THRESHOLD_WATTS = 20.0     # Threshold separating standby (<3W) from active
WM_CYCLE_MAX_GAP_MINUTES = 8         # Soak-phase bridging threshold
WM_CYCLE_MIN_DURATION_MINUTES = 15   # Minimum cycle duration
WM_CYCLE_MIN_PEAK_WATTS = 600.0      # Minimum required peak power
WM_CYCLE_MIN_ENERGY_KWH = 0.05       # Minimum cumulative energy

# Household Channel Mappings & Metadata
HOUSE_METADATA = {
    1: {"occupancy": 2, "type": "Detached, 4 bed", "washing_machine_col": "Appliance5", "wm_model": "Beko, WMC6140"},
    2: {"occupancy": 4, "type": "Semi-detached, 3 bed", "washing_machine_col": "Appliance2", "wm_model": "LG, F1289TD"},
    3: {"occupancy": 2, "type": "Detached, 3 bed", "washing_machine_col": "Appliance6", "wm_model": "Unknown"},
    20: {"occupancy": 2, "type": "Detached, 3 bed", "washing_machine_col": "Appliance4", "wm_model": "Unknown"},
}
```

---

# 4. Deep-Dive: Shared Data Utilities (`src/utils.py`)

### 4.1. Purpose & Responsibilities
`src/utils.py` contains the core gap-filling engine and the shared data loader. It guarantees that Section 2 (EDA) and Section 3 (Modeling) operate on the **identical preprocessing pipeline** directly from official cleaned data.

### 4.2. Algorithmic Breakdown: Custom Gap-Filling Engine
Standard `pandas.Series.interpolate(limit=N)` evaluates gaps greedily: in an outage of 1,000 minutes, it fills the first $N$ minutes and leaves the rest as `NaN`.

`fill_short_gaps(series, max_gap=3)` eliminates this flaw:
1. `get_nan_blocks(series)`: Locates all contiguous slices of `NaN` values, returning `(start_idx, end_idx, block_length)`.
2. For each block:
   - If `block_length <= max_gap`: It computes the step size between `left_val` and `right_val` and applies linear interpolation.
   - If `block_length > max_gap`: It **skips the block entirely**, leaving every single value as `NaN`.
3. `compute_outage_mask(series, max_gap=3)`: Evaluates original pre-interpolated blocks; any block with `block_length > max_gap` is marked `True` in a boolean mask `is_outage`.

```python
def fill_short_gaps(series: pd.Series, max_gap: int = 3) -> pd.Series:
    s = series.copy()
    blocks = get_nan_blocks(s)
    for start_pos, end_pos, length in blocks:
        if length <= max_gap:
            has_left = (start_pos > 0) and not np.isnan(s.iloc[start_pos - 1])
            has_right = (end_pos < len(s) - 1) and not np.isnan(s.iloc[end_pos + 1])
            if has_left and has_right:
                left_val = s.iloc[start_pos - 1]
                right_val = s.iloc[end_pos + 1]
                step = (right_val - left_val) / (length + 1)
                for step_idx, cur_pos in enumerate(range(start_pos, end_pos + 1), start=1):
                    s.iloc[cur_pos] = left_val + step * step_idx
            elif has_left and not has_right:
                s.iloc[start_pos:end_pos + 1] = s.iloc[start_pos - 1]
            elif not has_left and has_right:
                s.iloc[start_pos:end_pos + 1] = s.iloc[end_pos + 1]
    return s
```

### 4.3. The Shared Data Loader (`load_cleaned_house`)
1. Ingests `CLEAN_House{id}.csv` selecting `Time`, `Aggregate`, and the designated washing machine channel.
2. Filters out corrupted rows where `Issues == 1` (sub-meter sum > aggregate).
3. Averages concurrent sensor transmissions per second, then resamples onto uniform 1-minute intervals (`'1min'`).
4. Applies `fill_short_gaps(max_gap=3)` and returns the clean DataFrame along with the washing machine column name.
5. **Critical Guarantee**: Never imputes appliance missingness with zero.

---

# 5. Deep-Dive: Unit Testing Suite (`tests/test_gaps.py`)

### 5.1. Purpose & Verification
To prove that gap handling adheres to the strict specification, an automated unit test suite was built using `pytest`.

### 5.2. Test Cases Implemented
```python
def test_gap_handling_lengths():
    # 1. Gap of 1 minute -> MUST BE FILLED
    s1 = pd.Series([10.0, np.nan, 20.0])
    res1 = fill_short_gaps(s1, max_gap=3)
    assert not res1.isna().any()
    assert np.isclose(res1.iloc[1], 15.0)

    # 2. Gap of 2 minutes -> MUST BE FILLED
    s2 = pd.Series([10.0, np.nan, np.nan, 25.0])
    res2 = fill_short_gaps(s2, max_gap=3)
    assert not res2.isna().any()

    # 3. Gap of 3 minutes -> MUST BE FILLED
    s3 = pd.Series([10.0, np.nan, np.nan, np.nan, 30.0])
    res3 = fill_short_gaps(s3, max_gap=3)
    assert not res3.isna().any()

    # 4. Gap of 4 minutes -> MUST REMAIN FULLY NaN
    s4 = pd.Series([10.0, np.nan, np.nan, np.nan, np.nan, 30.0])
    res4 = fill_short_gaps(s4, max_gap=3)
    assert res4.isna().sum() == 4

    # 5. Gap of 10 minutes -> MUST REMAIN FULLY NaN
    s10 = pd.Series([10.0] + [np.nan] * 10 + [30.0])
    res10 = fill_short_gaps(s10, max_gap=3)
    assert res10.isna().sum() == 10
```
*Verification*: Run via `python -m pytest tests/test_gaps.py -v` $\rightarrow$ **100% Passed**.

---

# 6. Deep-Dive: Section 1 Raw Telemetry Inspection (`src/data_inspection.py`)

### 6.1. Ingestion of Multi-Gigabyte Raw Iterations
`src/data_inspection.py` reads `RAW_House1_Part1.csv` and `RAW_House1_Part2.csv` directly without altering data to discover sensor characteristics and transmission anomalies.

### 6.2. Empirical Inspection Findings (`raw_inspection_summary.json`)
* **Part 1 Rows**: 4,377,877 | **Start**: `2013-10-09 14:06:17` | **End**: `2014-10-01 01:27:07`
* **Part 2 Rows**: 4,140,886 | **Start**: `2014-06-27 21:57:50` | **End**: `2015-07-10 11:56:32`
* **Time vs Unix Clock Audit**: Verified `DateTime - Unix` offset is strictly 0.0 hours (confirming consistent UTC across all records).
* **Timestamp Monotonicity**: 1 backwards time jump in Part 1 (buffer jitter); strictly monotonic in Part 2.
* **Sampling Distribution ($\Delta t$)**: Median = 7.0s; 90th percentile = 8.0–15.0s; Max gap = 998.4 hours (~41.6 days) in Feb 2014.
* **Duplicate Timestamps**: 1,004,404 in Part 1 (22.94%); 14,755 in Part 2 (0.36%).
* **16-Bit Register Overflow**: Appliance 9 recorded maximum power of exactly **65,535 W** ($2^{16}-1$).
* **Appliance Spikes**: Spikes up to 98,301 W recorded on IAMs; Aggregate spikes up to 29,159 W.

---

# 7. Deep-Dive: Section 1 Raw Cleaning & Resampling Engine (`src/data_cleaning.py`)

### 7.1. Execution Flow
1. **Iteration Merging**: Switches from Part 1 to Part 2 at `2014-06-27 21:57:50`, resolving the 95.1-day overlap and preserving true `NaN` sub-meter dropouts.
2. **Plausibility Filtering**: Sets $P < 0\text{ W} \rightarrow \text{NaN}$, Aggregate $> 15{,}000\text{ W} \rightarrow \text{NaN}$, IAM $> 4{,}000\text{ W} \rightarrow \text{NaN}$.
3. **Deduplication**: Aggregates concurrent readings per second via arithmetic `mean()`, resolving 348,686 duplicate timestamps and compressing 7,260,898 merged rows to 6,912,212 unique seconds.
4. **1-Minute Uniform Resampling**: Resamples to `'1min'` using mean power aggregation (producing 920,031 intervals across 639 days).
5. **Robust Gap Policy**: Applies `fill_short_gaps(max_gap=3)` and flags outages $> 3\text{ minutes}$ in `is_outage = True`. Exactly **118,959 minutes (12.93%)** are flagged as genuine outages.
6. **Output Parquet**: Writes `output/House1_cleaned_1min.parquet` (10.9 MB).
7. **Validation Figure**: Generates `output/figures/sec1_raw_vs_cleaned.png` displaying unclipped raw vs 1-minute cleaned traces.

---

# 8. Deep-Dive: Section 2 Washing Machine EDA Engine (`src/eda_washing_machine.py`)

### 8.1. Multi-House Analysis Scope
Analyzes longitudinal 1-minute telemetry across four distinct REFIT homes using the official cleaned dataset via `load_cleaned_house()`:
* **House 1**: Channel 5 (`Appliance5`), Beko WMC6140, 2 occupants, Detached 4-bed.
* **House 2**: Channel 2 (`Appliance2`), LG F1289TD, 4 occupants, Semi-detached 3-bed.
* **House 3**: Channel 6 (`Appliance6`), Model Unknown, 2 occupants, Detached 3-bed.
* **House 20**: Channel 4 (`Appliance4`), Model Unknown, 2 occupants, Detached 3-bed.

---

# 9. Deep-Dive: Cycle Detection State Machine & Sensitivity Matrix

### 9.1. The Finite-State Machine with Soak-Phase Bridging
Washing machines draw power across four physical stages:
1. *Pulsed Heating*: Immersion heater draws ~2,000–2,400 W in 3–5 minute bursts.
2. *Agitation*: Bi-directional motor tumbling draws ~100–300 W.
3. *Soak Rest*: Power drops below 20 W for 5–8 minutes.
4. *Spin Extraction*: High-speed motor extraction draws ~400–600 W.

`detect_cycles()` maintains an active cycle state across soaking lulls:
* Power $\ge 20\text{ W}$ initiates or continues a cycle.
* Sub-threshold dips $\le 8\text{ minutes}$ are bridged into the ongoing cycle.
* Dips $> 8\text{ minutes}$ finalize the cycle block.
* Post-filtering enforces duration $\ge 15\text{ min}$ and peak $\ge 600\text{ W}$ (or energy $\ge 0.05\text{ kWh}$).

### 9.2. Cycle Detection Sensitivity Analysis
Evaluated across a $3 \times 3$ grid of power thresholds (10 W, 20 W, 50 W) and bridging gaps (5 min, 8 min, 12 min):

| Configuration | House 1 | House 2 | House 3 | House 20 |
| :--- | :---: | :---: | :---: | :---: |
| Th 10 W, Gap 5 min | 383 | 337 | 636 | 201 |
| Th 10 W, Gap 8 min | 376 | 332 | 627 | 191 |
| Th 10 W, Gap 12 min | 363 | 321 | 618 | 186 |
| Th 20 W, Gap 5 min | 386 | 335 | 637 | 203 |
| **Th 20 W, Gap 8 min (Baseline)** | **375** | **336** | **627** | **198** |
| Th 20 W, Gap 12 min | 363 | 327 | 618 | 188 |
| Th 50 W, Gap 5 min | 385 | 364 | 639 | 202 |
| Th 50 W, Gap 8 min | 371 | 343 | 631 | 202 |
| Th 50 W, Gap 12 min | 361 | 332 | 621 | 189 |

*Finding*: Cycle counts are highly stable. Increasing the bridging gap from 5 to 12 minutes consolidates multi-stage programs by ~5–7%; shifting the standby threshold from 10 W to 50 W has negligible effect because standby power is $< 3\text{ W}$.

---

# 10. Deep-Dive: Multi-Household Empirical Findings (1,536 Cycles)

### 10.1. Summary Statistics Table (`washing_machine_eda_summary.json`)
* **Extreme Sparsity**: Washing machines operate only **1.8% to 5.4% of the time**; they sit in off/standby mode **94.56% to 98.21%** of their operational life.
* **Duration Heterogeneity**: House 1 cycles are rapid (median 33.0 min; mean 41.5 min), whereas House 2 cycles are long and intensive (median 96.0 min; mean 98.0 min).
* **Energy Heterogeneity**: House 2 consumes more than double the energy per cycle (median 0.602 kWh) compared to House 1 (median 0.294 kWh).

### 10.2. Diurnal & Weekday Behavioral Patterns
* **Monday Peak**: **Monday is the heaviest laundry day** across all 4 homes: 298 cycles (**19.40%**).
* **Weekday Volume**: Tuesday (267 cycles, 17.38%), Wednesday (243 cycles, 15.82%), Friday (228 cycles, 14.84%), Thursday (213 cycles, 13.87%).
* **Weekend Minimum**: Saturday (141 cycles, **9.18%**) and Sunday (146 cycles, **9.51%**) are the lowest laundry days.
* **Morning Concentration**: **44.92% of all cycles (690 cycles)** start between **08:00 and 12:00**.
* **House 1 Overnight Washing**: Exactly **88 cycles (23.47% of House 1 cycles)** started between **00:00 and 06:00**, confirming automated delay-timer scheduling under UK Economy 7 off-peak tariffs.

---

# 11. Deep-Dive: Section 3 Offline Feature Engineering (`src/disaggregation_model.py`)

### 11.1. Contextual Receptive Field Formulation
To differentiate a 3-minute kettle spike from a 60-minute multi-stage wash cycle, the model requires symmetric temporal context. A 31-minute receptive field was engineered strictly from the aggregate load:
1. **Instantaneous Power**: $x(t)$
2. **Past Lags**: $x(t-1), x(t-2), x(t-3), x(t-5), x(t-10), x(t-15)$
3. **Future Leads**: $x(t+1), x(t+2), x(t+3), x(t+5), x(t+10), x(t+15)$
4. **Centered Rolling Statistics (Windows: 5, 15, 30, 60 min)**:
   - Rolling Mean ($\mu_w$): Identifies local power plateau.
   - Rolling Standard Deviation ($\sigma_w$): Distinguishes motor turbulence from smooth baseload.
   - Rolling Max & Min: Captures dynamic range.
5. **Rate-of-Change Edge Detectors**:
   - $\Delta_1 x(t) = x(t) - x(t-1)$
   - $\Delta_5 x(t) = x(t) - x(t-5)$
6. **Cyclical Temporal Encodings**:
   - $\text{hour\_sin} = \sin(2\pi \cdot \text{hour}/24)$, $\text{hour\_cos} = \cos(2\pi \cdot \text{hour}/24)$
   - $\text{dow\_sin} = \sin(2\pi \cdot \text{dow}/7)$, $\text{dow\_cos} = \cos(2\pi \cdot \text{dow}/7)$

---

# 12. Deep-Dive: Zero-Leakage Chronological Partitioning & Row Exclusion

### 12.1. Partition Timeline & Boundary Rules
* **Train Set (First 70%)**: `2013-10-09 13:06` to `2014-12-26 22:01` (644,063 total minutes)
* **Validation Set (Next 15%)**: `2014-12-26 22:02` to `2015-03-22 05:11` (138,014 total minutes)
* **Test Set (Final 15%)**: `2015-03-22 05:12` to `2015-07-10 11:56` (138,014 total minutes)

### 12.2. Row Exclusion Statistics (`extract_valid_dataset`)
Features were computed strictly within continuous segments. If any 31-minute window touched an outage or crossed a partition edge, that row produced a `NaN` feature and was dropped:
* **Train Set**: 105,519 rows excluded (16.38%) $\rightarrow$ **538,544 valid rows** (8,383 active minutes, 1.56%).
* **Validation Set**: 13,249 rows excluded (9.60%) $\rightarrow$ **124,765 valid rows** (2,327 active minutes, 1.87%).
* **Test Set**: 31,009 rows excluded (22.47%) $\rightarrow$ **107,005 valid rows** (2,162 active minutes, 2.02%).

---

# 13. Deep-Dive: Validation Set Tuning & The Subsampling Paradox

### 13.1. Sampling Strategy Benchmark (`validation_tuning_results.json`)
Because washing machines are idle $>95\%$ of the time, three training strategies were benchmarked on the Validation set:

| Strategy | Validation Overall MAE | Validation Active MAE | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Subsample 25% Inactive | 16.56 W | 442.86 W | 0.1886 | **0.9037** | 0.3121 |
| Class Weights (5x on Active) | 16.29 W | 456.01 W | 0.1812 | 0.8964 | 0.3015 |
| **Full Unweighted Training** | **13.06 W** | **500.86 W** | **0.3206** | 0.7241 | **0.4444** |

*The Subsampling Paradox Explained*: Subsampling the majority negative class caused the model to hallucinate false positive wash cycles on routine household cooking and heating surges, collapsing Precision to 18.8%. Full unweighted training taught the decision trees the true negative baseload, boosting Precision to 32.1% and F1 to 0.4444. Full data was frozen.

### 13.2. Postprocessing Zero-Cutoff Tuning
Evaluating zero-cutoffs from 0 W to 30 W on validation predictions revealed that a cutoff of **30.0 W** maximized F1-score (0.4654) and minimized overall MAE (12.65 W). Frozen choices: Cutoff = 30 W, Active Threshold = 20 W.

---

# 14. Deep-Dive: Model Architecture & HistGradientBoosting Selection

### 14.1. Why HistGradientBoostingRegressor?
1. **Handles Step-Discontinuities**: Resistive immersion heaters switch instantaneously from 0 W to 2,300 W. Tree-based axis-aligned splits model these step functions naturally without sigmoid smoothing.
2. **Training Speed & Efficiency**: Uses binned integer feature histograms (similar to LightGBM). Trains on 538,544 rows with 25 features in **under 20 seconds on a standard multi-core CPU**, eliminating GPU dependencies.
3. **Hyperparameters**: `max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `random_state=42`.

---

# 15. Deep-Dive: Final Within-Home Test Evaluation & Confusion Matrix

### 15.1. Test Evaluation Table (House 1: 107,005 Minutes)
Evaluated strictly on identical non-missing timestamps:

| Metric | Always-Off Baseline | Hour-of-Week Profile Baseline | Primary ML Model (HistGBM) |
| :--- | :---: | :---: | :---: |
| **Overall MAE (W)** | 10.40 W | 12.43 W | **5.91 W** |
| **Active MAE (W)** | 512.91 W | 509.75 W | **200.54 W** |
| **RMSE (W)** | 132.10 W | 132.12 W | **66.35 W** |
| **Precision** | 0.0000 | 0.0298 | **0.5453** |
| **Recall (On-State)**| 0.0000 | 0.0828 | **0.7512** |
| **Specificity** | **1.0000** | 0.9443 | **0.9871** |
| **F1-Score** | 0.0000 | 0.0438 | **0.6319** |
| **Ground Truth Energy** | 18.54 kWh | 18.54 kWh | 18.54 kWh |
| **Predicted Energy** | 0.00 kWh | 3.86 kWh | **15.32 kWh** |
| **Signed Energy Bias** | -18.54 kWh | -14.68 kWh | **-3.22 kWh** |
| **Energy Error (RETE %)**| 100.00% | 79.16% | **17.35%** |

### 15.2. Test Confusion Matrix Breakdown
* **True Negatives (TN)**: 103,489 minutes (**96.71%**)
* **False Positives (FP)**: 1,354 minutes (**1.27%**)
* **False Negatives (FN)**: 538 minutes (**0.50%**)
* **True Positives (TP)**: 1,624 minutes (**1.52%**)

### 15.3. The RETE Billing Myth Exposed
The Relative Error in Total Energy (RETE) is **17.35%** (-3.22 kWh bias on 18.54 kWh actual). RETE integrates power over time, allowing false positives and false negatives to cancel each other out. It measures aggregate volume error for grid planners, but must **never be described as billing accuracy**.

---

# 16. Deep-Dive: Leave-One-House-Out (LOHO) 4-Fold Cross-Validation

### 16.1. Empirical LOHO Results (`loho_cross_validation_results.json`)
Trained on 3 homes (~2 million samples) and evaluated on the unseen 4th home:

| Test Fold | Trained On | Test Samples | Model MAE | Always-Off MAE | Profile MAE | Model F1 | Profile F1 | Model Precision | Model Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **House 1** | Houses 2, 3, 20 | 770,373 | 11.72 W | 9.46 W | 18.75 W | **0.3040** | 0.0677 | 0.2149 | 0.5194 |
| **House 2** | Houses 1, 3, 20 | 652,422 | 24.08 W | 17.69 W | 23.31 W | **0.2417** | 0.1330 | 0.2127 | 0.2798 |
| **House 3** | Houses 1, 2, 20 | 696,834 | 44.40 W | 24.59 W | 27.14 W | **0.3615** | 0.1984 | 0.2819 | 0.5037 |
| **House 20** | Houses 1, 2, 3 | 618,239 | 9.68 W | 7.48 W | 16.42 W | **0.2307** | 0.1354 | 0.1972 | 0.2779 |

### 16.2. LOHO Takeaways
1. The ML model beats the Profile baseline in every single house.
2. Cross-household F1 drops from 0.63 to **0.23–0.36** due to appliance duration differences (House 1: 33 min vs House 2: 96 min) and distinct unmetered baseloads.
3. The Always-Off baseline minimizes overall MAE in all homes because machines operate $< 5\%$ of the time.

---

# 17. Deep-Dive: Quantitative Failure Mode Analysis

### 17.1. False Positives (Conflation with Resistive Loads)
* **Count**: 245 test minutes with prediction $\ge 200\text{ W}$ while washer was $< 20\text{ W}$.
* **Mean Aggregate Load**: **2,802.6 Watts**.
* **Root Cause**: Electric kettles (2.5–3.0 kW), dishwashers (2.0 kW), and electric ovens draw pure active power ($P$) profiles that mimic washing machine immersion heaters. Without reactive power ($Q$) or harmonics, active power alone conflates resistive heating elements.

### 17.2. False Negatives (Masking Under Concurrent Heavy Load)
* **Count**: 24 test minutes with washer $\ge 500\text{ W}$ while prediction $< 100\text{ W}$.
* **Mean Aggregate Load**: **3,085.8 Watts**.
* **Root Cause**: Heavy concurrent demand (cooking hobs, space heaters) masks the subtle motor tumble pulses of the washing machine inside massive background noise.

---

# 18. Visualizations Gallery & Detailed Interpretation (7 Figures)

### 18.1. Figure 1: Raw vs Cleaned Trace (`sec1_raw_vs_cleaned.png`)
* **Content**: 3-panel comparison (Aggregate, Fridge, Washing Machine) from 15 Nov 2013 to 18 Nov 2013.
* **Interpretation**: The gray unclipped raw trace displays sensor overflows up to 32 kW and sampling jitter. The solid colored lines demonstrate clean uniform 1-minute resampling: preserving fridge cyclic compressor pulses (~80–120 W) and the multi-stage wash cycle on 15 Nov (~2,400 W peak) without temporal distortion.

### 18.2. Figure 2: 24-Hour Whole-Home Trace (`sec2_24h_trace.png`)
* **Content**: 24-hour trace on active wash day **2015-02-23** in House 1.
* **Interpretation**: Overcomes the previous draft error (which plotted idle day 2013-11-17). Displays a genuine 106-minute wash cycle peaking at **2,367.8 W**. Other household spikes (1.5–2.0 kW) occur earlier and later in the day, demonstrating why magnitude alone cannot disaggregate loads.

### 18.3. Figure 3: Anatomy of a Wash Cycle (`sec2_detected_cycles.png`)
* **Content**: Zoomed wash cycle showing state machine start/end boundaries and phase annotations.
* **Interpretation**: Proves that water heating occurs in **three distinct pulsed bursts of 3–5 minutes each** (~2,200–2,400 W), followed by low-power motor tumbling (~150–250 W), an 8-minute soak rest (< 20 W), and high-speed spin extraction (~400–600 W).

### 18.4. Figure 4: Cross-Household Boxplots (`sec2_cross_house_comparison.png`)
* **Content**: Duration (min), Energy (kWh), and Peak Power (W) across Houses 1, 2, 3, and 20.
* **Interpretation**: Peak power is consistent across all homes (1,800–2,400 W) due to standard UK 230V heating elements. Duration and energy vary widely: House 1 median duration is 33.0 min (0.294 kWh) while House 2 median duration is 96.0 min (0.602 kWh).

### 18.5. Figure 5: Temporal Usage Distributions (`sec2_temporal_patterns.png`)
* **Content**: Bar charts of cycle start times by hour of day (0–23) and day of week across all 1,536 cycles.
* **Interpretation**: Proves Monday is the peak wash day (19.40%) and weekends are lowest (~9%). Confirms 44.92% morning peak between 08:00 and 12:00, and House 1 overnight off-peak washing (23.47%).

### 18.6. Figure 6: Test Actual vs. Predicted Trace (`sec3_actual_vs_predicted.png`)
* **Content**: Top panel aggregate load, bottom panel ground truth vs predicted washer load in May 2015.
* **Interpretation**: Shows model accurately locking onto high-power heating spikes and reconstructing the multi-stage envelope, while displaying brief false positive spikes during concurrent cooking events.

### 18.7. Figure 7: Confusion Matrix & Cumulative Energy (`sec3_confusion_matrix.png` & `sec3_cumulative_energy.png`)
* **Interpretation**:
  - `sec3_confusion_matrix.png`: Visualizes TN (103,489), FP (1,354), FN (538), and TP (1,624), documenting Precision (54.53%), Recall (75.12%), and Specificity (98.71%).
  - `sec3_cumulative_energy.png`: Tracks cumulative kWh over time. Demonstrates model (15.32 kWh) tracking Ground Truth (18.54 kWh) with a minor negative bias (-3.22 kWh), drastically outperforming the Profile baseline (3.86 kWh).

---

# 19. Deep-Dive: Section 4 Utility Strategy & M&V Design (`Recommendation.md`)

### 19.1. Commercial Demand-Side Management (DSM) Interventions
* **Opportunity**: 44.92% of cycles occur between 08:00 and 12:00; Monday is peak day (19.40%). Per-cycle energy varies >2x between homes (0.294 vs 0.602 kWh).
* **Intervention 1: Morning Peak Shifting**: Nudge customers running morning laundry to utilize appliance delay-timers to shift cycles to midday solar surplus hours (12:00–15:00) or off-peak hours.
* **Intervention 2: Eco-Cycle Targeting**: Target households averaging $>0.50\text{ kWh}$ per cycle with in-app tips on modern eco programs.
* **M&V Framework**: Randomized Controlled Trial (RCT) with 10,000 Treatment and 10,000 Control homes. Evaluated using a Difference-in-Differences (DiD) regression on peak-hour cycle initiation proportion:
  $$\Delta = (\bar{Y}_{\text{treat, post}} - \bar{Y}_{\text{treat, pre}}) - (\bar{Y}_{\text{ctrl, post}} - \bar{Y}_{\text{ctrl, pre}})$$
  Success metric: $\ge 15\%$ reduction in peak laundry starts.

---

# 20. Deep-Dive: Geographic Adaptation (UK vs. India) & 30-Min Meters

### 20.1. Why UK Models Fail in India
* **UK / Europe**: Overwhelmingly front-loading machines equipped with internal 2.0–2.4 kW immersion electric heaters.
* **India**: Overwhelmingly top-loading machines with pulsators or impellers. Critically, the vast majority **do not have internal electric water heaters**. They operate on cold water or draw hot water externally from gas/solar geysers. Their power profile consists strictly of **pulsed motor agitation drawing only 200 to 400 Watts**.
* **Failure Mechanism**: A model trained on UK REFIT data looks specifically for a 2 kW heating envelope. In India, it would produce a **~100% false negative rate**.

### 20.2. Impact of 15-Minute or 30-Minute Smart Meters
At 30-minute resolution, a 40-minute wash cycle collapses into 1 or 2 data points. A 2,000 W heating burst lasting 5 minutes averages out to $\Delta \bar{P} \approx 330\text{ W}$, becoming indistinguishable from microwaves, irons, or baseload drift. Minute-level time-series regression becomes mathematically underdetermined; utilities must transition to monthly probabilistic energy bounding (Factorial HMMs / Sparse Coding).

---

# 21. Pipeline Orchestrator & Interactive Notebook Walkthrough

### 21.1. `run_pipeline.py` Execution
* Pre-flight file verification checks that raw and cleaned CSVs exist.
* Executes Section 1 Inspection $\rightarrow$ Section 1 Cleaning $\rightarrow$ Section 2 EDA $\rightarrow$ Section 3 Modeling & LOHO.
* Total runtime: **~3.2 minutes on a standard CPU**.

### 21.2. `notebooks/refit_end_to_end_pipeline.ipynb`
* Formatted Jupyter notebook providing an interactive narrative walkthrough.
* Embeds all generated figures and tables for presentation to stakeholders.

---

# 22. AI Usage Transparency, Verification Protocol & Academic Citations

### 22.1. Tool Division of Labor
* **GitHub Copilot**: Inline syntax autocomplete and repetitive boilerplate (dictionary definitions, Matplotlib formatting, cyclical trigonometric formulas).
* **Claude / Antigravity**: Senior pair-programming auditor. Brainstormed the 31-minute receptive field, identified the pandas `interpolate(limit=3)` outage bug, and audited figure interpretations.
* **Author Verification**: Every script was executed locally; automated unit tests were run in pytest; every number and table was audited directly against raw JSON output files.

### 22.2. Key Academic Citations
* **Murray, D., Stankovic, L., & Stankovic, V.** (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. **Scientific Data**, 4, 160122.
* **Kelly, J., & Knottenbelt, W.** (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. In Proceedings of ACM BuildSys, 55–64.
* **Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M.** (2014). *NILMTK: An open source toolkit for non-intrusive load monitoring*. In Proceedings of ACM e-Energy, 265–276.
