import pandas as pd


# Rows below this efficiency ratio are flagged as candidate soiling days.
_ANOMALY_THRESHOLD = 0.85


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where efficiency_ratio falls below 0.85 (Section 2.7).

    These are candidate soiling days — days where actual generation dropped
    enough relative to the monthly median to suggest dust accumulation rather
    than normal weather variation.  The column `is_anomaly` is added to the
    DataFrame without modifying any existing columns.

    Args:
        df: Output of seasonal_baseline.compute_baseline — must contain an
            `efficiency_ratio` column.

    Returns:
        Copy of df with an extra boolean column `is_anomaly`.
    """
    if "efficiency_ratio" not in df.columns:
        raise ValueError("DataFrame must contain an 'efficiency_ratio' column. "
                         "Run seasonal_baseline.compute_baseline first.")

    result = df.copy()
    result["is_anomaly"] = result["efficiency_ratio"] < _ANOMALY_THRESHOLD
    return result
