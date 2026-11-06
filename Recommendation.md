# Recommendation: Utility Deployment & Energy Insights (Flock Energy)

**Target Audience:** Utility Operations, Product Strategy, and DSM Teams  
**Scope:** Washing Machine Disaggregation Feasibility, Customer Interventions, and Deployment Limits  

---

## 1. Executive Summary & Operational Viability
Using longitudinal smart-meter telemetry from the REFIT dataset, our offline disaggregation pipeline demonstrates that whole-home 1-minute aggregate power can reliably identify washing machine operation (within-home F1-score **0.6319**, recall **75.12%**, specificity **98.71%**). However, cross-household evaluation (LOHO) reveals that transfer performance drops to F1 **0.23–0.36** due to appliance and baseload heterogeneity.

### Suitable Operational Uses
1. **Aggregated Demand-Side Management (DSM)**: Forecasting community-level grid demand by tracking diurnal laundry distributions across thousands of homes.
2. **Retrospective Bill Decomposition**: Providing automated monthly energy breakdown estimates ("Where Your Energy Went") within offline billing cycles.
3. **Targeted Behavioral Nudges**: Identifying households with high morning or peak-hour laundry usage to encourage load shifting.

### Unsuitable Operational Uses
* **Real-Time Automated Circuit Control**: The offline model uses a 15-minute future lookahead; furthermore, false positives from concurrent resistive loads (kettles, dishwashers) would cause disruptive automated disconnections.
* **Statutory Sub-Billing**: A point-by-point precision of 54.53% and active MAE of 200.54 W do not meet legal utility metering standards.

---

## 2. Strategic Opportunity & M&V Framework

### Identified Opportunity: Morning Peak Shifting & High-Consumption Targeting
* **Empirical Grounding**: Across 1,536 detected cycles in 4 REFIT homes:
  * **Strong Morning Peak**: **44.92%** of all cycles start between **08:00 and 12:00**, with **Monday** being the heaviest wash day (**19.40%** of cycles).
  * **Wide Energy Variance**: Per-cycle energy varies by more than 2x between homes—from House 1 (median **0.294 kWh**) to House 2 (median **0.602 kWh**).
* **Intervention**:
  1. *Peak Shifting*: Nudge customers running laundry between 08:00–12:00 to shift usage to midday solar surplus hours (12:00–15:00) or off-peak hours, supported by appliance delay timers.
  2. *Eco-Cycle Nudge*: Target households exhibiting consistently high per-cycle energy (>0.50 kWh) with in-app tips on modern eco programs.
* **Measurement & Verification (M&V)**:
  * Conduct a Randomized Controlled Trial (RCT) across 10,000 treatment and 10,000 control homes.
  * Evaluate using a Difference-in-Differences (DiD) model on mean cycle energy and the proportion of cycles initiated in peak windows. Success metric: $\ge 15\%$ reduction in peak-hour cycle frequency.

---

## 3. Technology Limits & Data Granularity

### Technical Boundaries & Cross-Geography Transfer
1. **Appliance Conflation**: Single-phase active power cannot distinguish washing machine heating from electric kettles (~2.5–3 kW) or dishwashers, causing false spikes (mean aggregate during false positives: 2,802.6 W).
2. **Transfer to India**: UK machines rely on internal electric heating elements (~2.0–2.4 kW). Indian households predominantly use top-load pulsator machines with cold water or external geysers, drawing only 200–400 W of pulsed motor power. A model trained on UK heating spikes will fail completely in India without local retraining.

### Impact of 15-Minute / 30-Minute Smart-Meter Data
At 30-minute resolution, a typical 40-minute wash cycle collapses into only 1 or 2 averaged data points. Short pulsed heating stages (3–5 min) are diluted across the half-hour window ($\Delta \bar{P} \approx 300\text{ W}$), becoming indistinguishable from microwaves, irons, or baseload drift. True event-level disaggregation becomes unviable, necessitating aggregate monthly statistical estimation instead.
