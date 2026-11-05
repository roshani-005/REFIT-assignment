# REFIT Electrical Load Measurements: Cleaning and Appliance Disaggregation Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: REFIT](https://img.shields.io/badge/Dataset-REFIT_Smart_Home-green.svg)](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)

An end-to-end, reproducible data engineering, exploratory data analysis, and machine learning pipeline for Non-Intrusive Load Monitoring (NILM) appliance disaggregation, developed for the **Flock Energy Data Science Take-Home Assessment**.

---

## 1. What Was Built
This repository provides a modular, production-ready pipeline that processes real-world longitudinal smart-meter telemetry from the UK REFIT dataset:
1. **Raw Telemetry Ingestion & Cleaning Pipeline (`src/data_cleaning.py`)**:
   - Ingests multi-gigabyte raw telemetry (`RAW_House1_Part1.csv` & `Part2.csv`).
   - Audits timestamp monotonicity, duplicate timestamps, sampling intervals, and sensor error overflow codes (e.g. 65,535 W).
   - Merges iterations cleanly, deduplicates concurrent 1-second records via mean aggregation, resamples to uniform 1-minute resolution, interpolates short gaps ($\le 3$ min), and rigorously flags extended outages ($> 3$ min) without fabricating artificial data.
2. **Multi-Household Washing Machine EDA Engine (`src/eda_washing_machine.py`)**:
   - Analyzes longitudinal active washing machine power across four homes: **House 1, House 2, House 3, and House 20**.
   - Implements a domain-grounded finite-state cycle detection algorithm with soak-phase bridging.
   - Computes data quality metrics, standby percentages, and compares cycle characteristics (duration, peak power, energy per cycle) alongside diurnal and weekly usage patterns.
3. **Supervised Disaggregation Engine (`src/disaggregation_model.py`)**:
   - Formulates a 1-minute regression disaggregation model estimating washing machine active power from whole-home aggregate measurements.
   - Engineers a 31-minute contextual window of lagged features, centered rolling statistics, and edge-detection derivatives.
   - Evaluates a **HistGradientBoostingRegressor** against an **Always-Off Baseline** and a **Time-of-Week Profile Baseline** under strict non-leaking chronological validation.
   - Evaluates regression accuracy (MAE, Active MAE), classification quality (Precision, Recall, F1), and utility energy error (RETE %).

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
│   ├── data_inspection.py    # Section 1: Raw data checks & anomaly detection
│   ├── data_cleaning.py      # Section 1: Cleaning & 1-minute resampling pipeline
│   ├── eda_washing_machine.py# Section 2: Washing machine cycle detection & EDA
│   └── disaggregation_model.py# Section 3: Feature engineering, baselines, ML model
├── output/
│   ├── figures/              # Publication-grade figures
│   │   ├── sec1_raw_vs_cleaned.png
│   │   ├── sec2_24h_trace.png
│   │   ├── sec2_detected_cycles.png
│   │   ├── sec2_cross_house_comparison.png
│   │   ├── sec2_temporal_patterns.png
│   │   └── sec3_actual_vs_predicted.png
│   ├── raw_inspection_summary.json
│   ├── washing_machine_eda_summary.json
│   └── model_evaluation_metrics.json
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
* Python 3.10 or 3.11
* Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/refit-appliance-disaggregation.git
cd refit-appliance-disaggregation

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Placement
Ensure the following files are placed in the `data/` directory:
* Raw files in `data/raw/`:
  * `RAW_House1_Part1.csv`
  * `RAW_House1_Part2.csv`
* Cleaned files in `data/cleaned/`:
  * `CLEAN_House1.csv`
  * `CLEAN_House2.csv`
  * `CLEAN_House3.csv`
  * `CLEAN_House20.csv`

### Running the Complete Pipeline
To execute all stages from scratch (data inspection, cleaning, EDA, modeling, visualization):
```bash
python run_pipeline.py
```
To run the interactive Jupyter Notebook:
```bash
jupyter notebook notebooks/refit_end_to_end_pipeline.ipynb
```

---

## 4. Key Assumptions & Preprocessing Decisions

1. **Iteration Transition for House 1 Raw Data**:
   * Part 1 contains sensor drops recorded as `0`. Part 2 uses explicit `NaN` values. We transition from Part 1 to Part 2 on `2014-06-27 21:57:50` to preserve true `NaN` sensor metadata during the overlap.
2. **Duplicate Timestamp Resolution**:
   * Stated rule: Concurrent readings within the exact same second are aggregated via arithmetic `mean()`. This prevents data discarding and maintains instantaneous energy conservation.
3. **Gap Handling & Outage Boundary**:
   * Gaps $\le 3$ minutes are linearly interpolated (typical 1–2 packet transmission drops).
   * Gaps $> 3$ minutes are strictly preserved as `NaN` and flagged as `is_outage = True`. We do not impute long outages to avoid creating fictitious appliance energy.
4. **Active Cycle Definition**:
   * Minimum active threshold: $20\text{ W}$ (above 1–5 W standby).
   * Soak bridging: Pauses $\le 8$ minutes are merged into the ongoing cycle.
   * Plausibility filter: Duration $\ge 15$ minutes and peak power $\ge 600\text{ W}$ (or total energy $\ge 0.05\text{ kWh}$).

---

## 5. Validation Design & Leakage Prevention
* **Temporal Contiguity**: The validation design strictly rejects random train/test shuffling. 
* **Chronological Split**: House 1 data was partitioned into:
  * **Train (70%)**: Oct 2013 to Dec 2014 (561,848 minutes)
  * **Validation (15%)**: Dec 2014 to Mar 2015 (120,396 minutes)
  * **Test (15%)**: Mar 2015 to Jul 2015 (120,396 minutes)
* **Leakage Safeguards**: Rolling window feature engineering was computed within each partition without borrowing future information across partition boundaries. Preprocessing transformers were fit strictly on the training set.

---

## 6. Model Results & Trade-Offs

| Metric | Always-Off Baseline | Profile Baseline | HistGBM (Primary Model) | Assessment Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall MAE** | 10.61 W | 17.64 W | **8.81 W** | Outperforms all baselines. |
| **Active MAE** | 520.83 W | 505.34 W | **180.51 W** | **64.3% reduction** in active error. |
| **Recall (On-State)**| 0.00% | 32.75% | **94.48%** | Captures **94.5%** of actual wash minutes. |
| **F1-Score** | 0.0000 | 0.0769 | **0.4371** | Massive event-detection improvement. |
| **RETE (Energy Error)**| 100.00% | 27.38% | **19.95%** | Under **20%** total energy accounting error. |

* **Trade-Off Analysis**:
  * *Model Choice*: A `HistGradientBoostingRegressor` was selected over complex Deep Neural Networks (Seq2Point / CNNs). It offers near-identical tabular accuracy, executes in $< 60$ seconds, has zero GPU dependency, and is transparent and reproducible.
  * *Precision vs. Recall*: In demand-side management, missing an active laundry event (false negative) is more detrimental to peak load forecasting than minor false positives; the model intentionally prioritizes high recall (**94.48%**).

---

## 7. AI Tools Used
* **AI Assistance**: Antigravity (Google DeepMind) was used as a pair-programming assistant to brainstorm time-series feature engineering architectures, automate boilerplate pipeline scripting, and format Markdown reports. All mathematical formulations, physical thresholds, data validation pipelines, and analytical interpretations were verified and executed on the local environment.

---

## 8. Possible Improvements & Future Roadmap
1. **Factorial Hidden Markov Models (FHMM) or Semi-Markov Models**: Adding explicit state transitions (fill $\rightarrow$ heat $\rightarrow$ agitate $\rightarrow$ spin) to filter out single-point false positives caused by kettles.
2. **Deep Sequence-to-Point (Seq2Point) 1D-CNNs**: For larger multi-household transfer datasets, a 1D Dilated Temporal Convolutional Network can learn fine-grained temporal patterns directly from raw aggregate waveforms.
3. **Reactive Power & Phase Integration**: In future smart-meter rollouts, capturing reactive power ($Q$) and power factor would allow perfect discrimination between inductive motor loads and pure resistive heaters.

---

## 9. Academic Citations
* **Murray, D., Liao, J., Stankovic, L., Stankovic, V., Hauxwell-Baldwin, R., Wilson, C., Coleman, M., Kane, T., & Firth, S.** (2015). *A data management platform for personalised real-time energy feedback*. In Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting (EEDAL).
* **Kelly, J., & Knottenbelt, W.** (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. In Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments (BuildSys), 55–64.
* **Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M.** (2014). *NILMTK: An open source toolkit for non-intrusive load monitoring*. In Proceedings of the 5th International Conference on Future Energy Systems (e-Energy), 265–276.
* **Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C.** (2018). *Sequence-to-point learning with neural networks for nonintrusive load monitoring*. In Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
