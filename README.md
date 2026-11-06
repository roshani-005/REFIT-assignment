# REFIT Electrical Load Measurements: Cleaning and Appliance Disaggregation Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

[![Dataset: REFIT](https://img.shields.io/badge/Dataset-REFIT_Smart_Home-green.svg)](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)

An end-to-end, reproducible data engineering, exploratory data analysis, and machine learning pipeline for Non-Intrusive Load Monitoring (NILM) appliance disaggregation, developed for the **Flock Energy Data Science Take-Home Assessment**.

---

## 1. What Was Built
This repository provides an automated, modular pipeline that processes real-world longitudinal smart-meter telemetry from the UK REFIT dataset:
1. **Raw Telemetry Ingestion & Cleaning Pipeline (`src/data_cleaning.py`)**:
   - Ingests raw telemetry (`RAW_House1_Part1.csv` & `Part2.csv`).
   - Audits timestamp monotonicity, duplicate timestamps, sampling intervals, and sensor error overflow codes (e.g. 65,535 W).
   - Merges iterations cleanly, deduplicates concurrent 1-second records via mean aggregation, resamples to uniform 1-minute resolution, interpolates only short gaps ($\le 3$ min), and flags extended outages ($> 3$ min) without fabricating artificial data.
2. **Multi-Household Washing Machine EDA Engine (`src/eda_washing_machine.py`)**:
   - Analyzes longitudinal active washing machine power across four homes: **House 1, House 2, House 3, and House 20** using the official cleaned dataset.
   - Implements a domain-grounded finite-state cycle detection algorithm with soak-phase bridging.
   - Computes data quality metrics, standby percentages, and compares cycle characteristics (duration, peak power, energy per cycle) alongside diurnal and weekly usage patterns.
3. **Supervised Disaggregation Engine (`src/disaggregation_model.py`)**:
   - Formulates an **offline retrospective regression disaggregation model** estimating washing machine active power from whole-home aggregate measurements.
   - Engineers a 31-minute contextual window of lagged features, centered rolling statistics, and edge-detection derivatives within continuous valid segments.
   - Evaluates a **HistGradientBoostingRegressor** against an **Always-Off Baseline** and an **Hour-of-Week Profile Baseline** under strict non-leaking chronological validation and cross-household Leave-One-House-Out (LOHO) cross-validation.
   - Evaluates regression accuracy (MAE, Active MAE, RMSE), classification quality (Precision, Recall, Specificity, F1), and energy metrics (Signed Energy Bias, RETE %).

---

## 2. Repository Structure

```
refit_disaggregation/
├── data/
│   ├── raw/                  # RAW_House1_Part1.csv & Part2.csv
│   └── cleaned/              # CLEAN_House1.csv, House2.csv, House3.csv, House20.csv
├── src/
│   ├── __init__.py
│   ├── config.py             # Constants, house metadata, channel mappings
│   ├── utils.py              # Shared gap filling, outage masking, and data loader
│   ├── data_inspection.py    # Section 1: Raw data checks & anomaly detection
│   ├── data_cleaning.py      # Section 1: Cleaning & 1-minute resampling pipeline
│   ├── eda_washing_machine.py# Section 2: Washing machine cycle detection & EDA
│   └── disaggregation_model.py# Section 3: Feature engineering, baselines, ML model, LOHO
├── tests/
│   └── test_gaps.py          # Unit tests verifying short vs long gap handling
├── output/
│   ├── figures/              # Generated figures
│   │   ├── sec1_raw_vs_cleaned.png
│   │   ├── sec2_24h_trace.png
│   │   ├── sec2_detected_cycles.png
│   │   ├── sec2_cross_house_comparison.png
│   │   ├── sec2_temporal_patterns.png
│   │   ├── sec3_actual_vs_predicted.png
│   │   ├── sec3_confusion_matrix.png
│   │   └── sec3_cumulative_energy.png
│   ├── raw_inspection_summary.json
│   ├── washing_machine_eda_summary.json
│   ├── validation_tuning_results.json
│   ├── model_evaluation_metrics.json
│   ├── loho_cross_validation_results.json
│   └── failure_analysis.json
├── notebooks/
│   └── refit_end_to_end_pipeline.ipynb # Interactive notebook walkthrough
├── run_pipeline.py           # Master end-to-end execution script
├── requirements.txt          # Python environment dependencies
├── README.md                 # Project architecture, setup, and assumptions
├── DATA.md                   # Raw data quality findings and cleaning decisions
├── RESULTS.md                # Detailed EDA, model metrics, and failure modes
└── Recommendation.md         # Utility executive summary & Section 4 answers
```

---

## 3. Setup & Reproduction Instructions

### Prerequisites
* Python 3.11 (recommended)
* Git

### Installation
```bash
# Clone the repository
git clone https://github.com/roshani-005/REFIT-assignment.git
cd refit_disaggregation

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Download & Placement
Download the official REFIT dataset files and extract them into the following directory structure:
1. **Raw Files** in `data/raw/`:
   - `RAW_House1_Part1.csv`
   - `RAW_House1_Part2.csv`
2. **Cleaned Files** in `data/cleaned/`:
   - `CLEAN_House1.csv`
   - `CLEAN_House2.csv`
   - `CLEAN_House3.csv`
   - `CLEAN_House20.csv`

### Running Unit Tests
To verify that gap filling only interpolates runs $\le 3$ minutes and preserves longer runs as `NaN`:
```bash
python -m pytest tests/test_gaps.py -v
```

### Running the Complete Pipeline
To execute all stages end to end (raw inspection, cleaning, EDA, model training, validation tuning, test evaluation, LOHO cross-validation, and plot generation):
```bash
python run_pipeline.py
```

---

## 4. Key Assumptions & Decisions

1. **Iteration Transition for House 1 Raw Data**:
   * Part 1 contains sensor drops recorded as `0`. Part 2 uses explicit `NaN` values. We transition from Part 1 to Part 2 on `2014-06-27 21:57:50` to preserve true `NaN` sensor metadata during the overlap.
2. **Duplicate Timestamp Resolution**:
   * Concurrent readings within the exact same second are aggregated via arithmetic `mean()`.
3. **Strict Gap Handling & Outage Boundary**:
   * Gaps $\le 3$ minutes are linearly interpolated (typical 1–2 packet transmission drops).
   * Contiguous missing runs $> 3$ minutes are preserved strictly as `NaN` and flagged in `is_outage = True`.
4. **Active Cycle Definition**:
   * Active power threshold: $20\text{ W}$ (above 1–3 W standby).
   * Soak bridging: Pauses $\le 8$ minutes are merged into the ongoing cycle.
   * Duration $\ge 15$ minutes and peak power $\ge 600\text{ W}$ (or total energy $\ge 0.05\text{ kWh}$).
5. **Offline Inference Setting**:
   * The model incorporates a 31-minute contextual receptive field (including future aggregate readings from $t+1$ to $t+15$ and centered 60-minute rolling statistics). It is designed for offline retrospective energy auditing and billing decomposition, not online real-time stream processing.

---

## 5. Model Results (House 1 Unseen Test Set: 107,005 Minutes)

| Metric | Always-Off Baseline | Hour-of-Week Profile Baseline | HistGBM (Primary Model) |
| :--- | :---: | :---: | :---: |
| **Overall MAE** | 10.40 W | 12.43 W | **5.91 W** |
| **Active MAE** | 512.91 W | 509.75 W | **200.54 W** |
| **RMSE** | 132.10 W | 132.12 W | **66.35 W** |
| **Precision** | 0.0000 | 0.0298 | **0.5453** |
| **Recall (On-State)**| 0.0000 | 0.0828 | **0.7512** |
| **Specificity** | **1.0000** | 0.9443 | **0.9871** |
| **F1-Score** | 0.0000 | 0.0438 | **0.6319** |
| **Ground Truth Energy** | 18.54 kWh | 18.54 kWh | 18.54 kWh |
| **Predicted Energy** | 0.00 kWh | 3.86 kWh | **15.32 kWh** |
| **Signed Energy Bias** | -18.54 kWh | -14.68 kWh | **-3.22 kWh** |
| **RETE (Energy Error)**| 100.00% | 79.16% | **17.35%** |

*Note*: RETE is an aggregate energy error metric, not billing accuracy. Overpredictions and underpredictions partially cancel out over long integration windows, so point-by-point precision (54.53%) and active MAE (200.54 W) must be reviewed alongside total energy.

---

## 6. AI Tools Used
* **GitHub Copilot**: Used for inline code completion and repetitive syntax typing.
* **Claude / Antigravity**: Used for pair-programming assistance, code auditing, and critical review of figures, numbers, and interpolation logic.
* **Author Verification**: Every number, test script, mathematical formula, and data visualization in this repository was independently audited, executed, and verified by the author against generated JSON outputs.

---

## 7. Academic Citations
* **Murray, D., Stankovic, L., & Stankovic, V.** (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. **Scientific Data**, 4, 160122.  
  *Key takeaway: Foundational dataset publication establishing the 20-home longitudinal REFIT corpus and sensor specifications.*
* **Murray, D., Liao, J., Stankovic, L., Stankovic, V., Hauxwell-Baldwin, R., Wilson, C., Coleman, M., Kane, T., & Firth, S.** (2015). *A data management platform for personalised real-time energy feedback*. In Proceedings of EEDAL.  
  *Key takeaway: Demonstrates real-time energy feedback architectures and consumer engagement strategies using REFIT telemetry.*
* **Kelly, J., & Knottenbelt, W.** (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. In Proceedings of ACM BuildSys, 55–64.  
  *Key takeaway: Pioneered deep learning architectures (denoising autoencoders, recurrent nets) for residential appliance disaggregation.*
* **Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M.** (2014). *NILMTK: An open source toolkit for non-intrusive load monitoring*. In Proceedings of ACM e-Energy, 265–276.  
  *Key takeaway: Standardized NILM benchmarking metrics, dataset schemas, and baseline disaggregation algorithms.*
