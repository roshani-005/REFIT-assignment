# 3-Minute Technical Interview Pitch: REFIT Appliance Disaggregation Pipeline

**Speaker:** Candidate  
**Target Audience:** Technical Hiring Manager & Senior Data Scientists (Flock Energy)  
**Estimated Spoken Duration:** ~3 Minutes (~470 Words)  

---

### The Pitch (First-Person Script)

"Hello! I’m excited to share my end-to-end Non-Intrusive Load Monitoring—or NILM—pipeline built on the UK REFIT smart home dataset.

The core problem Flock Energy faces is residential peak demand. Major domestic wet appliances—specifically washing machines with internal immersion heaters—draw heavy two-kilowatt power spikes that place severe thermal stress on distribution transformers. While utilities want to offer personalized bill breakdowns and peak-shaving incentives, installing hardware sub-meters in every customer's home is economically unviable. My objective was to algorithmically disaggregate washing machine consumption directly from a single-point whole-home smart meter.

My approach was structured into three technical pillars:

First, in **Data Engineering**, I audited over 8.5 million raw telemetry readings. I resolved transmission duplicates by averaging concurrent one-second timestamps, eliminated 16-bit sensor overflow errors like 65,535 Watts, and addressed a critical pandas flaw. Standard `interpolate(limit=3)` inadvertently fills the start of long power cuts with fake energy. I engineered a custom block-length engine that interpolates only short sensor drops under three minutes while strictly preserving 118,959 minutes of genuine outages as NaNs.

Second, in **Exploratory Data Analysis**, I developed a finite-state machine to detect active wash cycles across four homes, covering 1,536 cycles. Because washing machines pause for five to eight minutes during fabric soaking, naive power thresholds artificially fragment a single laundry load into multiple mini-cycles. By implementing an eight-minute soak-phase bridge, I reconstructed accurate cycle boundaries. This revealed striking human behavior: laundry heavily peaks on **Monday mornings between 08:00 and 12:00 (accounting for 44.9% of all usage)**, while weekends are surprisingly the lowest wash days at around 9%. Furthermore, House 1 ran nearly a quarter of its laundry overnight, taking advantage of UK Economy 7 off-peak tariffs.

Third, in **Offline Modeling**, I engineered a 31-minute receptive field—spanning fifteen minutes of past lags, future leads, and centered rolling statistics—to give the model the temporal context needed to separate a quick kettle spike from a multi-stage wash program. To prevent data leakage, I enforced a strict chronological split and dropped window boundary rows touching outages. Using validation tuning, I proved that training on full unweighted data significantly reduced false positives compared to subsampling, boosting validation F1 to 0.44.

On the unseen House 1 test set, my **HistGradientBoostingRegressor** achieved an **F1-score of 0.6319**, a **Recall of 75.12%**, an **Active MAE of 200.54 Watts**—representing a 60% error reduction over baselines—and a **Relative Error in Total Energy of 17.35%**. I also conducted 4-fold **Leave-One-House-Out cross-validation**, finding that while the model beats time-of-week baselines across all unseen homes, cross-household F1 drops to 0.23 to 0.36 due to appliance variations.

Finally, I translated these findings into a utility DSM strategy: deploying a morning peak-shifting nudge program evaluated via an RCT and Difference-in-Differences framework, while establishing clear boundaries—such as why UK models fail in India where cold-water pulsators dominate, and how 30-minute smart meters require aggregate statistical bounding rather than minute-level regression.

I’d love to dive into any specific stage of the pipeline with you!"
