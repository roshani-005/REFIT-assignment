# Recommendation.md: Utility Executive Summary & Practical Implications

**Target Audience:** Utility Operations, Product Strategy, and Demand-Side Management Teams (Flock Energy)  
**Topic:** Operational Feasibility, Deployment Limits, and Customer Opportunities for Washing Machine NILM Disaggregation  

---

## 1. Executive Summary & Suitable Uses
Flock Energy’s smart-meter analytics pipeline aims to disaggregate major household loads from single-point whole-home measurements without installing intrusive sub-metering hardware. 

Based on empirical evaluation across longitudinal REFIT smart-meter telemetry, the developed 1-minute gradient-boosted disaggregation model achieves a **19.95% Relative Error in Total Energy (RETE)** and an **active-state recall of 94.48%**. 

### Suitable Operational Uses
1. **Aggregated Demand-Side Management (DSM) & Flexibility Forecasting**: Accurately tracking laundry start-time distributions by day and hour across thousands of homes allows distribution network operators (DNOs) to forecast peak distribution transformer loading.
2. **Personalized Customer Energy Audits & Bill Breakdown**: Disaggregating monthly washing machine energy enables automated monthly utility bills showing "Where Your Money Went" with ~80% billing-level accuracy.
3. **Time-of-Use (ToU) Tariff Nudging**: Pinpointing households running laundry during peak tariff windows (e.g. 16:00–19:00) to trigger targeted behavioral automated alerts.

### Unsuitable Operational Uses (Operational Boundaries)
* **Sub-Minute Real-Time Circuit Control**: The model is not designed for direct automated load-shedding relays, as occasional false positives (~639 minutes in test set) would disrupt customer laundry.
* **Legal Sub-Metering / Tenant Resale**: Inherent estimation uncertainty (~180 W active MAE) violates sub-billing certification standards.

---

## 2. Answers to Section 4 Questions

### 4.1. Practical Energy-Saving / Customer-Insight Opportunity
* **Identified Opportunity: "Cold-Water Wash Nudge & Dynamic Tariff Shifting"**
  * *Empirical Finding*: Our EDA revealed that wash cycle energy ranges widely—from **0.33 kWh** in House 1 (30°C/eco-mode) to **0.59 kWh** in House 2 (60°C/hot cotton cycles), with **85%+ of total wash energy consumed exclusively by the internal resistive water heater** (~2 kW element). Furthermore, 64% of laundry occurs between 08:00 and 14:00 or during evening peak hours.
  * *Customer Intervention*:
    1. **Eco-Wash Temperature Nudge**: Households identified with average cycle energy $> 0.55\text{ kWh}$ receive personalized mobile app insights: *"Switching your regular wash from 60°C to 30°C saves up to 45% electricity per load—saving £42/year without sacrificing cleanliness."*
    2. **Solar/Off-Peak Window Shifting**: Customers on Agile / ToU tariffs are nudged to utilize delay-timer functions to shift laundry to negative-pricing or high-solar periods (11:00–14:00).
  * *Measurement & Verification (M&V Methodology)*:
    * **Evaluation Framework**: A Randomized Controlled Trial (RCT) with a Treatment Group (10,000 households receiving nudges) and an uncontacted Control Group (10,000 households).
    * **Metric**: Difference-in-Differences (DiD) regression on:
      $$\Delta E_{\text{cycle}} = \big(\bar{E}_{\text{treatment, post}} - \bar{E}_{\text{treatment, pre}}\big) - \big(\bar{E}_{\text{control, post}} - \bar{E}_{\text{control, pre}}\big)$$
    * **Success Criteria**: A statistically significant ($\alpha=0.01$) reduction of $\ge 0.10\text{ kWh}$ per detected cycle and a $>15\%$ reduction in peak-hour wash frequency.

---

### 4.2. Core Limitations of the Technology

1. **Overlapping Appliance Signatures**:
   * Single-phase active power ($P$ in Watts) cannot uniquely identify a resistive element. An electric kettle boiling water (2,500 W for 3 minutes) or a dishwasher heating water (2,000 W) draw signatures identical to a washing machine's heating element, creating false positives.
2. **Telemetry Packet Loss & Missing Gaps**:
   * As demonstrated in House 1 raw data (12.79% outages, up to 41 days during network downtime), communications outages break temporal context. Disaggregation models cannot reliably reconstruct appliance duty cycles across multi-hour missing data blocks.
3. **Appliance & Household Heterogeneity**:
   * Drum capacities, internal heater wattage, eco-modes, and manufacturer programming vary drastically between homes (e.g. 41 min in House 1 vs 98 min in House 2). Models trained on one home lose accuracy when deployed across unseen homes without transfer learning.
4. **Cross-Geography Transferability (UK Homes to India)**:
   * **Water Heating Mechanisms**: UK machines are predominantly front-loaders with internal high-power immersion heaters (drawing ~2,000–2,400 W). In contrast, Indian households predominantly utilize **cold-water top-load machines with pulsators or impellers**, drawing only **200–400 W of pulsed motor load** without internal electric heating (or drawing hot water from separate gas/solar geysers).
   * **Failure of UK Models in India**: A model trained on UK REFIT data looks specifically for a 2,000 W thermal spike; when applied to an Indian home, it would fail completely (false negative rate $\approx 100\%$).
   * **Grid Quality & Backups**: Indian residential power frequently encounters voltage sags, brownouts, and inverter-battery (UPS) switches, altering electrical load profiles.

---

### 4.3. Impact of 15-Minute or 30-Minute Smart-Meter Data
In standard utility rollouts, smart meters only transmit 15-minute or 30-minute interval readings (kWh per interval).
* **Information Smeared Out**: When high-frequency active power is integrated over 30 minutes, distinct appliance transitions collapse. A 2,000 W washing machine heater running for 10 minutes produces an average boost of only:
  $$\Delta \bar{P}_{30\text{min}} = \frac{2000\text{ W} \times 10\text{ min}}{30\text{ min}} \approx 667\text{ W}$$
  This 667 W average is completely indistinguishable from a microwave running for 5 minutes, a clothes iron cycling, or continuous baseboard heating.
* **Loss of State Dynamics**: The distinctive temporal fingerprint of a wash cycle (fill $\rightarrow$ heat $\rightarrow$ agitate $\rightarrow$ soak $\rightarrow$ spin) spans multiple sub-states that last 2 to 8 minutes each. At 30-minute resolution, an entire 45-minute cycle collapses into just **1 or 2 blurred data points**. Fine-grained edge detection, cycle duration estimation, and multi-state classification become mathematically underdetermined without high-frequency waveform features.

---

## 3. Recommended Deployment Roadmap for Flock Energy
1. **Near-Term (1-Minute Telemetry Pilots)**: Deploy disaggregation for monthly customer engagement reports and aggregate grid peak forecasting.
2. **Mid-Term (30-Minute Telemetry Production)**: Switch algorithmic architecture from instantaneous regression to probabilistic aggregate daily energy bounding (e.g. Sparse Coding / Factorial Hidden Markov Models) to estimate monthly washer consumption rather than minute-by-minute power.
3. **Geographic Adaptation**: Collect a localized 100-home pilot dataset in India before deploying any disaggregation models outside the UK market.
