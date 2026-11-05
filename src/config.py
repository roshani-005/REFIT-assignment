"""
REFIT Appliance Disaggregation Configuration and Metadata
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Appliance mappings from CLEAN_READ_ME_081116.txt
# Channel indexing: Aggregate is col 2, Appliance1 is col 3, ..., Appliance9 is col 11
HOUSE_METADATA = {
    1: {
        "washing_machine_col": "Appliance5",
        "fridge_col": "Appliance1",
        "dishwasher_col": "Appliance6",
        "occupancy": 2,
        "type": "Detached, 4 bed",
        "wm_model": "Beko, WMC6140",
    },
    2: {
        "washing_machine_col": "Appliance2",
        "fridge_col": "Appliance1",
        "dishwasher_col": "Appliance3",
        "occupancy": 4,
        "type": "Semi-detached, 3 bed",
        "wm_model": "LG, F1289TD",
    },
    3: {
        "washing_machine_col": "Appliance6",
        "fridge_col": "Appliance2",
        "dishwasher_col": "Appliance5",
        "occupancy": 2,
        "type": "Detached, 3 bed",
        "wm_model": "Unknown",
    },
    20: {
        "washing_machine_col": "Appliance4",
        "fridge_col": "Appliance1",
        "dishwasher_col": "Appliance5",
        "occupancy": 2,
        "type": "Detached, 3 bed",
        "wm_model": "Unknown",
    },
}

# Cleaning & Resampling Constants
RESAMPLE_INTERVAL = "1min"
MAX_INTERPOLATION_MINUTES = 3  # short gaps <= 3 mins are interpolated
POWER_UPPER_LIMIT_AGG = 15000.0  # W (residential circuit breaker threshold)
POWER_UPPER_LIMIT_IAM = 4000.0   # W (standard UK socket rating 13A * 240V ~ 3120W, with margin)

# Washing Machine Cycle Detection Parameters
WM_ACTIVE_THRESHOLD_WATTS = 20.0     # Power above standby (standby is typically 1-5W)
WM_CYCLE_MAX_GAP_MINUTES = 8         # Soak / pause phases merged into same cycle
WM_CYCLE_MIN_DURATION_MINUTES = 15   # Real wash cycle minimum length
WM_CYCLE_MIN_PEAK_WATTS = 600.0      # At least one heating/spin phase spike
WM_CYCLE_MIN_ENERGY_KWH = 0.05       # Minimum cumulative energy for a valid cycle
