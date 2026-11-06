# DATA.md: REFIT Raw Data Inspection and Cleaning Report

## 1. Executive Summary
This document provides a comprehensive audit of the raw electrical load measurements from the REFIT Smart Home dataset (specifically **House 1**), details the cleaning and resampling pipeline developed for Section 1, and documents all data quality decisions, gap handling strategies, and residual limitations.

---

## 2. Raw Dataset Overview (House 1)
The raw dataset for House 1 is divided into two sequential iterations reflecting changes in the data collection infrastructure:
* **`RAW_House1_Part1.csv`**: Covers **2013-10-09 14:06:17** to **2014-10-01 01:27:07** (4,377,877 rows). In this iteration, unavailable/unresponsive sub-meter sensors were recorded as `0`.
* **`RAW_House1_Part2.csv`**: Covers **2014-06-27 21:57:50** to **2015-07-10 11:56:32** (4,140,886 rows). In this iteration, unavailable sensors were properly recorded as `NaN` to distinguish non-operational sensors from true zero power consumption.
* **Overlap Period**: Between **2014-06-27 21:57:50** and **2014-10-01 01:27:07** (95.1 days), both files record measurements concurrently. Our pipeline resolves this by using Part 1 prior to `2014-06-27 21:57:50` and transitioning cleanly to Part 2 thereafter to exploit explicit `NaN` values.

---

## 3. Data Quality Findings & Anomalies

### 3.1. Timestamp Parsing & Monotonicity
* **Timestamp Format**: Datetimes are formatted as `%Y-%m-%d %H:%M:%S` alongside Unix epoch timestamps.
* **Time vs Unix Offset Audit**: Verified that `DateTime - Unix` offset is strictly 0.0 hours throughout both Part 1 and Part 2, confirming consistent UTC timestamps without daylight saving shifts.
* **Monotonicity**:
  * In `Part 1`, **1 backwards step** was identified where the sensor clock buffered and transmitted out of sequence.
  * In `Part 2`, timestamps were strictly monotonically non-decreasing (0 backwards steps).

### 3.2. Duplicate Timestamps
* **`RAW_House1_Part1.csv`**: Contains **1,004,404 duplicate timestamps** (**22.94%** of all rows). These duplicates occur because multiple sensor channels transmitted within the same second or the server polling script polled asynchronously every 6–8 seconds.
* **`RAW_House1_Part2.csv`**: Contains **14,755 duplicate timestamps** (**0.36%** of all rows).

### 3.3. Sampling Interval Distribution ($\Delta t$)
The telemetry uses an event-based / periodic polling mechanism nominally reporting every 6–8 seconds:
| Metric | RAW Part 1 | RAW Part 2 | Interpretation |
| :--- | :--- | :--- | :--- |
| **Median $\Delta t$** | **7.0 seconds** | **7.0 seconds** | Consistent nominal sensor polling rate. |
| **50th Percentile** | 7.0 seconds | 7.0 seconds | Standard steady-state interval. |
| **90th Percentile** | 15.0 seconds | 8.0 seconds | Moderate delays due to transmission retries. |
| **99th Percentile** | 16.0 seconds | 9.0 seconds | Tail latency before dropping packets. |
| **Max Gap** | **998.4 hours** (~41.6 days) | **207.6 hours** (~8.6 days) | Severe network/hardware outage in Feb 2014. |

### 3.4. Gap Breakdown
* **Normal polling ($\Delta t \le 10\text{s}$)**:
  * Part 1: 1,904,024 intervals (43.5%)
  * Part 2: 4,098,393 intervals (99.0%)
* **Short delays ($10\text{s} < \Delta t \le 180\text{s}$)**:
  * Part 1: 1,469,439 intervals (33.6%)
  * Part 2: 27,646 intervals (0.7%)
* **Extended Outages ($\Delta t > 180\text{s}$ / 3 minutes)**:
  * Part 1: 462 distinct outages
  * Part 2: 91 distinct outages

### 3.5. Sensor Overflows & Impossible Power Spikes
Electrical domestic wiring in the UK operates on single-phase 230V/240V AC. Standard ring mains are protected by 32A breakers (~7.4 kW), with standard 13A wall sockets (~3.1 kW). The main household service fuse is typically 60A–100A (~14 kW to 23 kW).
Analysis of raw sub-meters and aggregate load revealed clear transmission and register overflow errors:
* **Appliance 9**: Max value recorded was **65,535 W**. This is precisely $2^{16} - 1$, representing a classic 16-bit unsigned integer sensor error code.
* **Appliance 1, 2, 4, 7, 8**: Recorded spikes between **65,549 W** and **98,301 W** (physically impossible for standard domestic appliances).
* **Aggregate Load**: Recorded spikes up to **29,159 W** (exceeding residential main service entrance capacity).
* **Negative Power**: No negative values were recorded ($P \ge 0\text{ W}$ across all channels).

---

## 4. Documented Cleaning & Resampling Pipeline

The cleaning pipeline (`src/data_cleaning.py`) performs the following sequential transformations:

1. **Iteration Merging**:
   * Part 1 is ingested for timestamps up to `2014-06-27 21:57:50`.
   * Part 2 is ingested from `2014-06-27 21:57:50` onwards, ensuring true sensor dropouts are maintained as `NaN` rather than spurious zeros.
2. **Chronological Sorting**:
   * All records are sorted by parsed datetime to resolve any buffering jitter.
3. **Plausibility & Outlier Filtering**:
   * Values exceeding physical thresholds are set to `NaN`:
     * Whole-home Aggregate $> 15{,}000\text{ W}$
     * Individual Appliance Monitors (IAMs) $> 4{,}000\text{ W}$
     * Values $< 0\text{ W}$
4. **Duplicate Deduplication**:
   * Concurrent readings sharing the exact same timestamp second are aggregated via arithmetic **`mean()`**. This averages concurrent sub-second state transitions and avoids discarding valid power measurements.
   * Deduplication reduced 7,260,898 merged raw records to **6,912,212 unique 1-second timestamps** (348,686 duplicate records resolved).
5. **Uniform 1-Minute Resampling**:
   * Resampled to a regular 1-minute grid (`'1min'`) using mean aggregation:
     $$\bar{P}_{1\text{min}} = \frac{1}{N} \sum_{i=1}^N P(t_i)$$
   * Total 1-minute intervals produced across the 639-day window: **920,031 minutes**.
6. **Strict Gap Handling & Outage Flagging Policy**:
   * **Flawed Approach Replaced**: Standard `series.interpolate(limit=N)` was rejected because pandas interpolates the first $N$ minutes of an arbitrarily long outage (e.g. turning the first 3 minutes of a 40-day outage into false data).
   * **True Gap Filling Implementation**: We implemented `fill_short_gaps(series, max_gap=3)` which evaluates the *entire length* of contiguous missing blocks. Only blocks of total length $\le 3$ minutes are linearly interpolated; blocks $> 3$ minutes are left strictly untouched as `NaN`.
   * **Outage Masking**: `compute_outage_mask(series, max_gap=3)` evaluates original NaN blocks prior to interpolation. Contiguous gaps $> 3$ minutes are flagged in a boolean column `is_outage = True`.
   * **Outage Statistics**: Exactly **118,959 minutes** (**12.93%** of the timeline) were flagged as genuine outages and unfillable gaps.

---

## 5. Visual Validation (Before vs. After Cleaning)
The cleaning pipeline output was visually validated over a 3-day representative window (`2013-11-15` to `2013-11-18`) in `output/figures/sec1_raw_vs_cleaned.png`:
* **Before Cleaning**: The raw uncleaned trace displays raw unclipped measurements, directly exposing sensor register overflows and erratic bursts.
* **After Cleaning**:
  * **Aggregate Load**: Shows smooth regular 1-minute load profiles without high-frequency sensor jitter or false spikes.
  * **Appliance 1 (Fridge)**: Regular cyclic compressor pulses (~80–120 W, 20-minute duty cycles) are cleanly preserved on the 1-minute grid.
  * **Appliance 5 (Washing Machine)**: The distinctive multi-stage wash cycle maintains its exact shape and energy area without temporal distortion.

---

## 6. Remaining Data Concerns & Limitations
1. **Unmonitored Baseload**: The 9 IAM channels only monitor a subset of home appliances. Unmetered circuits (lighting, cooking hobs, electric showers, immersion heaters) remain embedded in the aggregate signal, creating background noise for NILM models.
2. **Missing Sub-meter Metadata in Part 1**: Because Part 1 set unavailable sub-meters to 0 rather than NaN, cold periods where a sensor was unplugged cannot be distinguished from a switched-off appliance prior to June 2014.
3. **Single Active Power Measurement**: The dataset records only real active power ($P$ in Watts). The lack of reactive power ($Q$ in VAR), apparent power ($S$), or power factor limits the ability to distinguish pure resistive heaters (kettles) from inductive motor loads (washers) of identical wattage.
