# Comprehensive Technical Interview Defense & Q&A Guide

**Project:** REFIT Smart Home Appliance Disaggregation (NILM)  
**Target Enterprise:** Flock Energy  
**Format:** Categorized Question-and-Answer Defense Guide with Technical Justifications  

---

## Table of Contents
1. [Architecture & System Design](#1-architecture--system-design)
2. [Data Engineering, Gap Filling & Pandas Gotchas](#2-data-engineering-gap-filling--pandas-gotchas)
3. [Electromechanical Domain Physics & Cycle Detection](#3-electromechanical-domain-physics--cycle-detection)
4. [Feature Engineering & The 31-Minute Receptive Field](#4-feature-engineering--the-31-minute-receptive-field)
5. [Data Leakage Prevention & Window Integrity](#5-data-leakage-prevention--window-integrity)
6. [Validation Set Tuning & The Subsampling Paradox](#6-validation-set-tuning--the-subsampling-paradox)
7. [Evaluation Metrics & The "RETE Billing" Myth](#7-evaluation-metrics--the-rete-billing-myth)
8. [Cross-Household Generalization (LOHO)](#8-cross-household-generalization-loho)
9. [Failure Mode Analysis & Physical Conflation](#9-failure-mode-analysis--physical-conflation)
10. [Utility Strategy, Geographies (UK vs. India) & 30-Min Meters](#10-utility-strategy-geographies-uk-vs-india--30-min-meters)
11. [Engineering Difficulties & How They Were Solved](#11-engineering-difficulties--how-they-were-solved)
12. [AI Disclosure & Verification Defense](#12-ai-disclosure--verification-defense)

---

## 1. Architecture & System Design

### Q1.1: Why did you choose an OFFLINE retrospective architecture instead of an online real-time streaming model?
**Answer:**  
"I deliberately framed this pipeline as an **offline retrospective disaggregation system** for two primary reasons:
1. **Utility Use-Case Alignment**: Flock Energy’s primary commercial drivers are end-of-month billing decomposition ('Where Your Energy Went' reports), retrospective carbon audits, and community-level Demand-Side Management (DSM) peak forecasting. None of these use cases require sub-second latency.
2. **Mathematical Power of Centered Context**: Domestic wash cycles are multi-stage electromechanical processes spanning 40 to 100 minutes. To distinguish an initial 2,200 W heating burst of a washing machine from a 2,500 W tea kettle, the model requires future context. By utilizing a symmetric 31-minute receptive field ($t-15$ to $t+15$) and centered 60-minute rolling statistics, the model can observe whether the power spike is followed by motor tumbling or immediately returns to baseload. 

Calling a model with a 15-minute future lookahead 'real-time' would be scientifically dishonest; framing it as offline retrospective reflects real production engineering."

### Q1.2: Why did you use Parquet format for cleaned telemetry instead of CSV?
**Answer:**  
"Cleaned House 1 comprises 920,031 one-minute rows across 11 numerical power channels. 
1. **I/O Performance & Storage**: CSV storage is uncompressed and requires expensive ASCII-to-float parsing on every load (~85 MB). Parquet uses columnar dictionary encoding and Snappy compression, shrinking storage to 10.9 MB (an ~8x compression ratio) and loading in under 0.8 seconds.
2. **Schema & Null Preservation**: Parquet natively preserves data types (`float32`, `datetime64[ns]`, `bool`) and distinguishes between `NaN` and missingness without string parsing ambiguities."

---

## 2. Data Engineering, Gap Filling & Pandas Gotchas

### Q2.1: Why is standard `series.interpolate(limit=N)` considered a severe bug in smart meter telemetry, and how did you solve it?
**Answer:**  
"This was one of the most critical bugs I audited. In Pandas, `series.interpolate(method='time', limit=3)` does not check the overall length of an outage—it indiscriminately fills the first $N$ missing intervals of *any* gap, regardless of whether the gap lasts 4 minutes or 40 days.

In House 1, there was a major network outage in February 2014 lasting 998.4 hours (~41.6 days). If you call `interpolate(limit=3)`, pandas will fabricate 3 minutes of fictitious power data at the start of that 41-day outage. If the last recorded reading before the outage was a 2,000 W heater, you just injected 100 Watt-hours of non-existent energy into the database.

**My Solution (`src/utils.py`)**:  
I wrote a custom block-length evaluator (`fill_short_gaps`):
1. It calculates the contiguous run-length of every NaN block.
2. If and only if the entire block length is $\le 3\text{ minutes}$ (representing routine sensor packet drops), it applies linear interpolation.
3. If the block length is $> 3\text{ minutes}$ (representing a true power cut or router outage), it preserves the entire block as `NaN`.
4. I verified this logic with automated unit tests in [`tests/test_gaps.py`](file:///C:/Users/yrosh/.gemini/antigravity/scratch/refit_disaggregation/tests/test_gaps.py), proving that 1, 2, and 3-minute gaps are interpolated, while 4 and 10-minute gaps strictly remain `NaN`."

### Q2.2: What was the $65,535\text{ W}$ anomaly you discovered in the raw data?
**Answer:**  
"In `RAW_House1_Part1.csv`, Appliance 9 recorded maximum power values of exactly **65,535 Watts**. 

In domestic UK electrical systems, households operate on single-phase 230V protected by a 60A–100A main fuse (~14 kW to 23 kW), and single appliances plug into 13A wall sockets (~3 kW). A reading of 65,535 W is physically impossible. 

Mathematically, $65{,}535 = 2^{16} - 1$, which is the maximum integer representable by an unsigned 16-bit binary register (`uint16`). This was an uncaught firmware crash/overflow code sent by the transmitter when the sensor clipped. I handled this by applying a physical sanity filter: setting Aggregate $> 15{,}000\text{ W}$ and IAMs $> 4{,}000\text{ W}$ to `NaN`."

### Q2.3: How did you handle the 95-day overlap between Part 1 and Part 2 of House 1?
**Answer:**  
"`RAW_House1_Part1.csv` ends on `2014-10-01`, but `RAW_House1_Part2.csv` begins earlier, on `2014-06-27` (a 95.1-day overlap). 

The critical difference is metadata quality: in Part 1, when sub-meter transmitters lost connection, the logging software recorded them as `0 W`. In Part 2, the logging infrastructure was upgraded to properly record disconnected sensors as `NaN`. 

If you use Part 1 during the overlap, you mistake disconnected sensors for appliances that are turned off. Therefore, my pipeline strictly ingests Part 1 prior to `2014-06-27 21:57:50` and transitions immediately to Part 2 thereafter, maximizing explicit missingness preservation."

---

## 3. Electromechanical Domain Physics & Cycle Detection

### Q3.1: How does a washing machine physically consume power, and why did you reject the claim that it has a continuous 12-minute heating phase?
**Answer:**  
"A domestic washing machine has three primary electrical loads:
1. An internal resistive immersion heater (~2,000–2,400 W).
2. A drive motor (~100–300 W during agitation, ~400–600 W during high-speed spinning).
3. Electronics, drain pump, and water inlet solenoids (< 50 W).

When inspecting zoomed high-resolution traces in House 1 (`sec2_detected_cycles.png`), I found that water heating does **not** occur as a single uninterrupted 12-minute block. Instead, it fires in **three distinct pulsed bursts of ~3–5 minutes each**, separated by drum rotation. The heating element pulses to allow the rotating drum to circulate water and distribute heat evenly across the wash load without scorching fabrics. Describing it as a flat 12-minute block misrepresents the appliance's thermal dynamics."

### Q3.2: Why is 'Soak Bridging' necessary in your cycle detection algorithm?
**Answer:**  
"A washing machine program includes a 'soak/rest' phase where the drum stops rotating to allow detergent enzymes to break down stains. During this stage, active power drops below 20 W for 5 to 8 consecutive minutes.

If you use a simple power threshold ($\ge 20\text{ W}$), your algorithm will prematurely declare the cycle finished when soaking begins, and then declare a brand new cycle when the motor resumes spinning! This artificially inflates cycle counts by 200–300% and drastically underestimates cycle duration.

My state machine incorporates an **8-minute soak bridge**: sub-threshold dips lasting $\le 8\text{ minutes}$ are bridged into the ongoing cycle. In our sensitivity analysis, extending this bridging parameter beyond 8 minutes produced stable cycle counts across all 4 homes."

### Q3.3: What was the most surprising behavioral pattern you discovered in the EDA?
**Answer:**  
"The most counter-intuitive finding was the **weekday distribution**:
* Popular intuition assumes laundry peaks heavily on weekends (Saturday and Sunday).
* In reality across 1,536 detected cycles, **Monday was the single highest laundry day (19.40%)**, followed by Tuesday (17.38%), while **Saturday (9.18%) and Sunday (9.51%) were the lowest wash days of the entire week!**
* Furthermore, **44.92%** of all cycles started in the morning between **08:00 and 12:00**.
* In House 1, **23.47%** of cycles started between midnight and 06:00, showing that the occupants programmed delay timers to take advantage of cheaper off-peak Economy 7 night tariffs."

---

## 4. Feature Engineering & The 31-Minute Receptive Field

### Q4.1: If an aggregate meter reads 2,500 Watts at 12:00 PM, why can't a model know if the washing machine is on from that number alone?
**Answer:**  
"Because power is a scalar quantity that sums additively across all active circuits. 2,500 W could be:
* An electric kettle boiling water (2,500 W for 2–3 minutes).
* An electric shower heating water (8,000 W clipped or running on half-power).
* An electric oven element cycling on (2,200 W).
* A washing machine water heater (2,300 W).

At a single timestamp $t$, a kettle and a washing machine look identical. The only way to differentiate them is **temporal duration and profile shape**: a kettle shuts off after 2 minutes; a washing machine continues into motor tumbling pulses for 45 minutes. That is why temporal context is mandatory."

### Q4.2: What features did you engineer, and why?
**Answer:**  
"I engineered a 31-minute contextual receptive field strictly from the aggregate load:
1. **Lags & Leads ($t-15$ to $t+15$)**: Captures the immediate waveform envelope before and after timestamp $t$.
2. **Centered Rolling Statistics (5, 15, 30, 60 min)**:
   - *Rolling Mean*: Establishes local average baseline.
   - *Rolling Standard Deviation*: Detects motor turbulence vs steady baseload.
   - *Rolling Max/Min*: Captures dynamic range.
3. **Rate of Change ($\Delta_1 x, \Delta_5 x$)**: Acts as a step-edge detector, pinpointing the exact second a heating element switches on.
4. **Cyclical Temporal Encodings**: $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$ to represent time smoothly without a artificial discontinuity between 23:59 and 00:00."

---

## 5. Data Leakage Prevention & Window Integrity

### Q5.1: Why is random train/test splitting catastrophic in time-series NILM?
**Answer:**  
"In standard tabular machine learning, you randomly shuffle rows. In time-series electricity telemetry, **random shuffling causes catastrophic target leakage due to high temporal autocorrelation**.

If timestamp 12:01 PM is in the training set and 12:02 PM is in the test set, the model does not learn appliance physics—it simply memorizes that the appliance was on 60 seconds earlier. This produces fake 99% accuracy benchmarks that crash completely in production.

I enforced a **strict chronological split** on House 1:
* Train: First 70% of timeline (Oct 2013 to Dec 2014)
* Validation: Next 15% of timeline (Dec 2014 to Mar 2015)
* Test: Final 15% of timeline (Mar 2015 to Jul 2015)
The model was evaluated on completely unseen future seasons."

### Q5.2: How did you prevent rolling windows from leaking across outages and partition boundaries?
**Answer:**  
"If you calculate a 60-minute centered rolling window at the start of your Test set, a naive implementation will look 30 minutes backward into the Validation set. That is data leakage. Furthermore, if a window spans across a 5-day power cut, calculating a rolling mean across the gap fabricates an impossible average.

**My Solution (`extract_valid_dataset`)**:  
I engineered features within each partition independently. Any timestamp whose 31-minute receptive field touched an outage ($>3$ min NaN) or extended beyond the partition edge produced a `NaN` feature and was strictly dropped:
* Train: 16.38% rows excluded $\rightarrow$ 538,544 valid rows.
* Validation: 9.60% rows excluded $\rightarrow$ 124,765 valid rows.
* Test: 22.47% rows excluded $\rightarrow$ 107,005 valid rows.
This guaranteed that every single sample evaluated possessed a continuous, valid 31-minute window."

---

## 6. Validation Set Tuning & The Subsampling Paradox

### Q6.1: What was the 'Subsampling Paradox' you uncovered during validation tuning?
**Answer:**  
"Because washing machines are off $>95\%$ of the time, the dataset suffers from severe class imbalance. A common heuristic in machine learning is to downsample the majority negative class (e.g., keeping all active minutes and sampling 25% of inactive minutes) to balance the data and speed up training.

On the Validation set, I formally benchmarked three strategies:
1. *Subsampling 25% Inactive*: F1 = 0.3121, Precision = 0.1886, Recall = 0.9037.
2. *Class Weights (5x)*: F1 = 0.3015, Precision = 0.1812, Recall = 0.8964.
3. *Full Unweighted Training*: **F1 = 0.4444, Precision = 0.3206**, Recall = 0.7241.

**Why Subsampling Failed**:  
When you discard 75% of inactive minutes, the model never learns what ordinary high-power household activity looks like (e.g. baking in an oven or running an iron). As a result, the subsampled model hallucinated false positives on every household power spike, tanking precision down to 18.8%. Training on the full unweighted dataset taught the decision trees the true negative baseload, boosting Precision to 32.1% and F1 to 0.4444. I froze the Full Data strategy based on this empirical proof."

---

## 7. Evaluation Metrics & The "RETE Billing" Myth

### Q7.1: Why does the Always-Off baseline achieve an Overall MAE of 10.40 W, and why is that metric misleading?
**Answer:**  
"In House 1, the washing machine operates only ~1.8% of the time. If a baseline model predicts $0\text{ W}$ for every single minute of the year, it is 100% correct 98.2% of the time! That is why the Always-Off baseline achieves a seemingly impressive Overall MAE of 10.40 W.

However, its Active MAE is **512.91 W**, its Recall is **0.0%**, and its F1-score is **0.0**. 

Evaluating NILM on Overall MAE alone is an analytical trap. You must evaluate **Active MAE** (error when the appliance is actually running) and **F1-Score / Precision / Recall** to measure whether the model provides any actual utility."

### Q7.2: Why must Relative Error in Total Energy (RETE) never be described as 'billing accuracy'?
**Answer:**  
"Relative Error in Total Energy is defined as:
$$\text{RETE} = \frac{|\sum \hat{y} - \sum y|}{\sum y} \times 100\%$$

In my House 1 test evaluation, the model achieved a RETE of **17.35%** (predicting 15.32 kWh vs. actual 18.54 kWh).

Calling this '82.6% billing accuracy' is misleading because RETE integrates power over time, allowing **false positives and false negatives to cancel each other out**. If the model misses a 5 kWh laundry cycle on Tuesday (false negative) but mistakes a 5 kWh dishwasher cycle for a washing machine on Thursday (false positive), the total energy error is 0%, even though point-by-point accuracy is completely wrong! 

RETE is a useful aggregate metric for transformer capacity planning, but it must always be reported alongside point-by-point Precision (54.53%) and Recall (75.12%)."

---

## 8. Cross-Household Generalization (LOHO)

### Q8.1: Why did you run Leave-One-House-Out (LOHO) cross-validation, and what did it reveal?
**Answer:**  
"Within-home validation (training on House 1 past, testing on House 1 future) tells you how well a model learns a specific household's habits and machine. But Flock Energy needs to know: *Can a model trained on existing homes disaggregate a brand new customer's home without installing sensors first?*

I ran a 4-fold Leave-One-House-Out cross-validation across Houses 1, 2, 3, and 20:
* House 1 Test: F1 = 0.3040 (vs. Profile Baseline F1 = 0.0677)
* House 2 Test: F1 = 0.2417 (vs. Profile Baseline F1 = 0.1330)
* House 3 Test: F1 = 0.3615 (vs. Profile Baseline F1 = 0.1984)
* House 20 Test: F1 = 0.2307 (vs. Profile Baseline F1 = 0.1354)

**Core Insights**:
1. In every single house, the ML model substantially outperformed the Time-of-Week Profile baseline.
2. However, cross-household F1 dropped from 0.63 (within-home) to **0.23–0.36**. This gap reflects **appliance heterogeneity**: House 1 uses a fast 33-minute Beko machine consuming 0.29 kWh, while House 2 uses a 96-minute LG machine consuming 0.60 kWh. A universal model struggles to generalize across radically different machine programs without calibration."

---

## 9. Failure Mode Analysis & Physical Conflation

### Q9.1: Why did your model produce 245 minutes of False Positives in the test set?
**Answer:**  
"In my failure mode analysis, I found that during the 245 false positive minutes, the average household aggregate load was **2,802.6 Watts**. 

The fundamental root cause is **physical signature conflation**. In single-phase active power telemetry ($P$ in Watts), a pure resistive heating element looks identical regardless of what appliance contains it. An electric kettle boiling water draws 2,500–3,000 W; a dishwasher on a wash cycle draws 2,000 W; an electric shower draws 7,000 W. 

Because standard utility meters record only active real power ($P$) and lack reactive power ($Q$, VAR), harmonics, or power factor, the model occasionally misclassifies concurrent kettle boiling or dishwashing as a washing machine heating spike."

### Q9.2: What caused the 24 minutes of False Negatives?
**Answer:**  
"The false negatives ($y_{\text{true}} \ge 500\text{ W}$ while $\hat{y} < 100\text{ W}$) occurred when the average household aggregate load was extraordinarily high and erratic—averaging **3,085.8 Watts**. 

When multiple heavy appliances operate concurrently (e.g. electric cooking hob plus immersion space heating), the subtle drum-rotation pulses (~150 W) of the washing machine become submerged in background electrical noise, masking the cycle."

---

## 10. Utility Strategy, Geographies (UK vs. India) & 30-Min Meters

### Q10.1: How would you design a commercial Demand-Side Management (DSM) program for Flock Energy based on your findings?
**Answer:**  
"I grounded my strategy directly in the empirical data:
1. **The Peak Shifting Opportunity**: 44.92% of all cycles occur between 08:00 and 12:00, with Monday being the highest laundry day (19.40%). 
2. **The Intervention**: Deploy automated mobile app notifications nudging customers running morning laundry to utilize appliance delay-timers to shift cycles into midday solar surplus hours (12:00–15:00) or overnight off-peak windows.
3. **M&V Framework**: Evaluate via a **Randomized Controlled Trial (RCT)**:
   - 10,000 Treatment households (receiving automated nudges).
   - 10,000 Control households (receiving standard service).
   - Evaluate using a **Difference-in-Differences (DiD)** regression on the proportion of cycles initiated during peak windows:
   $$\Delta = (\bar{Y}_{\text{treat, post}} - \bar{Y}_{\text{treat, pre}}) - (\bar{Y}_{\text{ctrl, post}} - \bar{Y}_{\text{ctrl, pre}})$$
   Success metric: A statistically significant ($\alpha=0.01$) reduction of $\ge 15\%$ in peak-hour cycle initiation."

### Q10.2: Why would a washing machine model trained in the UK completely fail if deployed in India?
**Answer:**  
"Because of fundamental differences in domestic water heating infrastructure:
* **UK / European Machines**: Overwhelmingly front-loading machines equipped with internal high-power electric immersion heating elements (drawing 2,000–2,400 W).
* **Indian Domestic Machines**: Overwhelmingly top-loading machines with pulsators or impellers. Critically, the vast majority **do not have internal electric water heaters**. They operate on cold water or draw hot water externally from separate gas or solar geysers. Their power profile consists strictly of **pulsed motor agitation drawing only 200 to 400 Watts**.

A machine learning model trained on UK REFIT data learns that a washing machine is characterized by a 2 kW heating envelope. If deployed in India, it would look for a 2 kW surge that never happens, resulting in a **100% false negative rate**. Localized data collection is mandatory."

### Q10.3: What happens if Flock Energy only has access to 15-minute or 30-minute smart meter data?
**Answer:**  
"At 30-minute resolution, high-frequency electromechanical features are destroyed:
1. A 40-minute wash cycle collapses into **only 1 or 2 averaged data points**.
2. A 2,000 W heating burst lasting 5 minutes averages out over 30 minutes to:
   $$\Delta \bar{P} = \frac{2000\text{ W} \times 5\text{ min}}{30\text{ min}} \approx 333\text{ W}$$
   This 333 W rise is completely indistinguishable from an electric iron, a microwave running for 3 minutes, or baseline refrigerator drift.
3. *Engineering Implication*: Minute-level time-series regression becomes mathematically underdetermined. The utility must transition from instantaneous power tracking to **probabilistic monthly energy bounding** (e.g. Factorial Hidden Markov Models or Sparse Coding) to estimate monthly aggregate kWh rather than individual cycle starts."

---

## 11. Engineering Difficulties & How They Were Solved

### Q11.1: What was the hardest data synchronization challenge you faced?
**Answer:**  
"Handling transmission jitter and asynchronous polling. The raw REFIT sensors polled every 6 to 8 seconds, generating over 1 million duplicate timestamps where multiple sub-meters reported in the same second. 

If you take `first()` or `last()`, you discard valid state transitions and violate instantaneous energy conservation. I solved this by implementing an arithmetic `mean()` aggregation per second, compressing 7.26 million raw records to 6.91 million unique 1-second timestamps before resampling onto the continuous 1-minute grid."

---

## 12. AI Disclosure & Verification Defense

### Q12.1: Exactly how did you use GitHub Copilot versus Claude / Antigravity?
**Answer:**  
"I maintained strict division of labor and full transparency:
1. **GitHub Copilot**: Acted as an **inline syntax autocomplete tool**. It predicted repetitive dictionary definitions in `src/config.py`, cyclical trigonometric formulas in feature engineering, and standard Matplotlib subplot formatting.
2. **Claude / Antigravity**: Acted as a **senior pair-programming auditor**. It assisted in brainstorming feature architectures (e.g. designing the 31-minute receptive field), critically reviewing figures, and identifying the flaw in pandas `interpolate(limit=3)`.
3. **Author Verification (My Role)**: Every script was executed locally on my machine. I wrote and executed the automated unit test suite in pytest, and personally audited every single metric, table, and graph in my documentation directly against raw output JSON files to ensure zero hallucination."
