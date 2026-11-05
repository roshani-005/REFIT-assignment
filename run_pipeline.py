"""
End-to-End Execution Pipeline for REFIT Data Cleaning and Appliance Disaggregation
Runs all steps from raw inspection to model evaluation and visualization.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_inspection import run_inspection
from src.data_cleaning import clean_raw_house1
from src.eda_washing_machine import run_eda
from src.disaggregation_model import train_and_evaluate


def main():
    start_time = time.time()
    print("=" * 70)
    print("  REFIT DATA CLEANING & APPLIANCE DISAGGREGATION: END-TO-END RUN")
    print("=" * 70)
    
    # Step 1: Raw Data Inspection
    print("\n[STEP 1/4] Inspecting Raw House 1 Data...")
    run_inspection()
    
    # Step 2: Raw Data Cleaning Pipeline
    print("\n[STEP 2/4] Running Raw House 1 Cleaning & Resampling Pipeline...")
    clean_raw_house1()
    
    # Step 3: Washing Machine EDA Across 4 Cleaned Homes
    print("\n[STEP 3/4] Running Washing Machine EDA Across 4 Cleaned Homes...")
    run_eda()
    
    # Step 4: Disaggregation Model Training & Evaluation
    print("\n[STEP 4/4] Training & Evaluating Disaggregation Model...")
    train_and_evaluate()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed/60.0:.2f} MINUTES")
    print("  All figures and summary metrics saved to output/ and output/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
