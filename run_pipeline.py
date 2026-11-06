"""
End-to-End Execution Pipeline for REFIT Data Cleaning and Appliance Disaggregation
Runs all 4 sections in order:
1. Raw House 1 Inspection (Section 1)
2. Raw House 1 Cleaning & Parquet Output (Section 1)
3. Washing Machine EDA across Houses 1, 2, 3, 20 (Section 2)
4. Offline Appliance Disaggregation & LOHO Evaluation (Section 3)
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import RAW_DATA_DIR, CLEAN_DATA_DIR, HOUSE_METADATA
from src.data_inspection import run_inspection
from src.data_cleaning import clean_raw_house1
from src.eda_washing_machine import run_eda
from src.disaggregation_model import main as run_disaggregation


def verify_prerequisites():
    """
    Checks that all required raw and cleaned REFIT dataset files exist.
    Stops with a helpful message if files are missing.
    """
    required_raw = [
        RAW_DATA_DIR / "RAW_House1_Part1.csv",
        RAW_DATA_DIR / "RAW_House1_Part2.csv",
    ]
    required_clean = [
        CLEAN_DATA_DIR / f"CLEAN_House{hid}.csv"
        for hid in [1, 2, 3, 20]
    ]
    
    missing_files = [str(f) for f in required_raw + required_clean if not f.exists()]
    
    if missing_files:
        print("\n" + "!" * 70)
        print("ERROR: Missing required REFIT dataset files:")
        for mf in missing_files:
            print(f"  - {mf}")
        print("\nPlease download the official REFIT dataset and place the CSV files into:")
        print(f"  - Raw data:     {RAW_DATA_DIR}")
        print(f"  - Cleaned data: {CLEAN_DATA_DIR}")
        print("!" * 70 + "\n")
        sys.exit(1)
        
    print("[OK] All required raw and cleaned data files verified.")


def main():
    start_time = time.time()
    print("=" * 70)
    print("  REFIT DATA CLEANING & APPLIANCE DISAGGREGATION: END-TO-END RUN")
    print("=" * 70)
    
    verify_prerequisites()
    
    # Step 1: Raw Data Inspection
    print("\n[STEP 1/4] Inspecting Raw House 1 Data (Timestamps, Clamping, Overlaps)...")
    run_inspection()
    
    # Step 2: Raw Data Cleaning Pipeline
    print("\n[STEP 2/4] Running Raw House 1 Cleaning & 1-Minute Resampling Pipeline...")
    clean_raw_house1()
    
    # Step 3: Washing Machine EDA Across 4 Cleaned Homes
    print("\n[STEP 3/4] Running Washing Machine EDA Across 4 Cleaned Homes (1, 2, 3, 20)...")
    run_eda()
    
    # Step 4: Disaggregation Model Training & Evaluation
    print("\n[STEP 4/4] Training & Evaluating Offline Disaggregation Model & LOHO...")
    run_disaggregation()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed/60.0:.2f} MINUTES")
    print("  All figures and summary metrics saved to output/ and output/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
