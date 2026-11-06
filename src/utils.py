"""
Shared utilities for REFIT data handling.
Provides:
1. fill_short_gaps: Fills ONLY contiguous NaN runs of length <= max_gap.
   Never touches runs of length > max_gap (unlike pandas interpolate(limit=N)).
2. compute_outage_mask: Flags timestamps belonging to original gaps > max_gap.
3. load_cleaned_house: Single shared loader for official cleaned REFIT CSVs,
   used identically by both EDA and disaggregation modeling.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_DIR, HOUSE_METADATA, RESAMPLE_INTERVAL, MAX_INTERPOLATION_MINUTES


def get_nan_blocks(series: pd.Series) -> List[Tuple[int, int, int]]:
    """
    Returns list of (start_idx, end_idx, length) for each contiguous run of NaNs.
    Indices are integer position-based (0 to len(series)-1).
    end_idx is inclusive.
    """
    is_na = series.isna().values
    blocks = []
    in_block = False
    start_pos = 0
    
    for pos, na in enumerate(is_na):
        if na:
            if not in_block:
                in_block = True
                start_pos = pos
        else:
            if in_block:
                in_block = False
                blocks.append((start_pos, pos - 1, pos - start_pos))
                
    if in_block:
        blocks.append((start_pos, len(is_na) - 1, len(is_na) - start_pos))
        
    return blocks


def fill_short_gaps(series: pd.Series, max_gap: int = MAX_INTERPOLATION_MINUTES) -> pd.Series:
    """
    Linearly interpolates ONLY contiguous NaN blocks whose total length is <= max_gap.
    Blocks with length > max_gap are left completely as NaN (zero points in the block are filled).
    This strictly satisfies the requirement to flag longer outages rather than filling them
    without justification, avoiding the pandas interpolate(limit=N) bug which fills the first N
    points of long outages.
    """
    s = series.copy()
    blocks = get_nan_blocks(s)
    n = len(s)
    
    for start_pos, end_pos, length in blocks:
        if length <= max_gap:
            # Check if we have valid bounding points on both sides for linear interpolation
            has_left = (start_pos > 0) and not np.isnan(s.iloc[start_pos - 1])
            has_right = (end_pos < n - 1) and not np.isnan(s.iloc[end_pos + 1])
            
            if has_left and has_right:
                left_val = s.iloc[start_pos - 1]
                right_val = s.iloc[end_pos + 1]
                # Linear interpolation across length steps
                # step = (right_val - left_val) / (length + 1)
                step = (right_val - left_val) / (length + 1)
                for step_idx, cur_pos in enumerate(range(start_pos, end_pos + 1), start=1):
                    s.iloc[cur_pos] = left_val + step * step_idx
            elif has_left and not has_right:
                # Trailing short gap: forward fill
                s.iloc[start_pos:end_pos + 1] = s.iloc[start_pos - 1]
            elif not has_left and has_right:
                # Leading short gap: backward fill
                s.iloc[start_pos:end_pos + 1] = s.iloc[end_pos + 1]
        else:
            # Length > max_gap: strictly preserve as NaN
            pass
            
    return s


def compute_outage_mask(series: pd.Series, max_gap: int = MAX_INTERPOLATION_MINUTES) -> pd.Series:
    """
    Returns a boolean Series (same index) where True indicates that the timestamp
    belongs to a contiguous NaN block whose total length > max_gap.
    Must be computed on the ORIGINAL series before any interpolation.
    """
    mask = pd.Series(False, index=series.index)
    blocks = get_nan_blocks(series)
    
    for start_pos, end_pos, length in blocks:
        if length > max_gap:
            mask.iloc[start_pos:end_pos + 1] = True
            
    return mask


def load_cleaned_house(house_id: int,
                       max_gap: int = MAX_INTERPOLATION_MINUTES,
                       clean_data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, str]:
    """
    Single shared loader for the OFFICIAL cleaned REFIT dataset.
    Used identically by Section 2 (EDA) and Section 3 (Disaggregation Modeling).
    
    Steps:
    1. Reads CLEAN_House{id}.csv (Time, Aggregate, washing_machine_col, Issues).
    2. Parses DateTime and discards corrupted rows where Issues == 1.
    3. Aggregates concurrent timestamps per second via mean().
    4. Resamples onto regular 1-minute grid ('1min').
    5. Applies fill_short_gaps(max_gap) and computes per-channel outage masks.
       Never fills appliance NaNs with zero; keeps per-channel missingness.
    """
    data_dir = clean_data_dir or CLEAN_DATA_DIR
    csv_path = data_dir / f"CLEAN_House{house_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned REFIT file not found: {csv_path}")
        
    meta = HOUSE_METADATA[house_id]
    wm_col = meta["washing_machine_col"]
    
    cols_to_read = ['Time', 'Aggregate', wm_col]
    
    # Peek header to check for Issues column
    header_df = pd.read_csv(csv_path, nrows=2)
    has_issues = 'Issues' in header_df.columns
    if has_issues:
        cols_to_read.append('Issues')
        
    df = pd.read_csv(csv_path, usecols=cols_to_read)
    df['DateTime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime')
    
    if has_issues:
        valid_mask = (df['Issues'] == 0)
        df = df[valid_mask]
        
    # Group by second to resolve duplicate sensor transmissions, then resample to 1-minute
    power_cols = ['Aggregate', wm_col]
    for c in power_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df_sec = df.groupby('DateTime')[power_cols].mean()
    df_1min = df_sec.resample(RESAMPLE_INTERVAL).mean()
    
    # Apply gap policy per channel
    outage_df = pd.DataFrame(index=df_1min.index)
    for c in power_cols:
        outage_df[f"{c}_is_outage"] = compute_outage_mask(df_1min[c], max_gap=max_gap)
        df_1min[c] = fill_short_gaps(df_1min[c], max_gap=max_gap)
        
    df_1min['is_outage'] = outage_df['Aggregate_is_outage'] | outage_df[f"{wm_col}_is_outage"]
    
    return df_1min, wm_col
