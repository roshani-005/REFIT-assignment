"""
Unit tests for gap handling functions:
Verifies that fill_short_gaps:
- Correctly fills gaps of length 1, 2, 3 minutes with linear interpolation
- Leaves gaps of length 4 and 10 minutes completely as NaN (never fills first N minutes)
- Correctly identifies outage masks
"""

import numpy as np
import pandas as pd
import pytest

from src.utils import fill_short_gaps, compute_outage_mask, get_nan_blocks


def test_gap_handling_lengths():
    # Construct a test series with known gap lengths:
    # 0: 10.0
    # 1: NaN (gap length 1)
    # 2: 20.0
    # 3: NaN
    # 4: NaN (gap length 2)
    # 5: 30.0
    # 6: NaN
    # 7: NaN
    # 8: NaN (gap length 3)
    # 9: 40.0
    # 10: NaN
    # 11: NaN
    # 12: NaN
    # 13: NaN (gap length 4 -> MUST NOT BE FILLED)
    # 14: 50.0
    # 15..24: 10 NaNs (gap length 10 -> MUST NOT BE FILLED)
    # 25: 60.0
    
    vals = (
        [10.0, np.nan, 20.0] +
        [np.nan, np.nan, 30.0] +
        [np.nan, np.nan, np.nan, 40.0] +
        [np.nan, np.nan, np.nan, np.nan, 50.0] +
        [np.nan] * 10 + [60.0]
    )
    idx = pd.date_range("2020-01-01 00:00:00", periods=len(vals), freq="1min")
    s = pd.Series(vals, index=idx)
    
    # Check NaN block detection
    blocks = get_nan_blocks(s)
    lengths = [b[2] for b in blocks]
    assert lengths == [1, 2, 3, 4, 10], f"Expected blocks [1,2,3,4,10], got {lengths}"
    
    # Compute outage mask (max_gap=3)
    outage_mask = compute_outage_mask(s, max_gap=3)
    # Gaps of length 1, 2, 3 should NOT be outages
    assert not outage_mask.iloc[1]
    assert not outage_mask.iloc[3:5].any()
    assert not outage_mask.iloc[6:9].any()
    # Gaps of length 4 and 10 MUST be flagged as outages
    assert outage_mask.iloc[10:14].all()
    assert outage_mask.iloc[15:25].all()
    
    # Fill short gaps
    filled = fill_short_gaps(s, max_gap=3)
    
    # 1-minute gap filled: midpoint of 10 and 20 is 15
    assert not np.isnan(filled.iloc[1])
    assert np.isclose(filled.iloc[1], 15.0)
    
    # 2-minute gap filled: between 20 and 30 (step = 10/3 = 3.333...)
    assert not filled.iloc[3:5].isna().any()
    assert np.isclose(filled.iloc[3], 20.0 + 10.0 / 3.0)
    assert np.isclose(filled.iloc[4], 20.0 + 20.0 / 3.0)
    
    # 3-minute gap filled: between 30 and 40 (step = 10/4 = 2.5)
    assert not filled.iloc[6:9].isna().any()
    assert np.isclose(filled.iloc[6], 32.5)
    assert np.isclose(filled.iloc[7], 35.0)
    assert np.isclose(filled.iloc[8], 37.5)
    
    # 4-minute gap: MUST REMAIN COMPLETELY NaN!
    # (Unlike pandas interpolate(limit=3) which would fill the first 3 values)
    assert filled.iloc[10:14].isna().all(), "All 4 minutes of a 4-minute gap must remain NaN!"
    
    # 10-minute gap: MUST REMAIN COMPLETELY NaN!
    assert filled.iloc[15:25].isna().all(), "All 10 minutes of a 10-minute gap must remain NaN!"
    
    print("All gap handling tests passed successfully!")


if __name__ == "__main__":
    test_gap_handling_lengths()
