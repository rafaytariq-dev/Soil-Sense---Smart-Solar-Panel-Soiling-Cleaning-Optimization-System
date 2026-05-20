import pandas as pd


def compute_baseline(csv_path: str) -> pd.DataFrame:
    """Compute monthly median generation baseline from the FoxESS daily report.

    Loads `solar_report_6months.csv`, groups by calendar month, computes the
    median daily generation (kWh) for each month, then annotates every row
    with the expected generation and an efficiency_ratio (Section 2.7).

    efficiency_ratio = actual_generation / monthly_median_generation
        > 1.0  — above-average day
        ~ 1.0  — normal
        < 0.85 — candidate soiling day (flagged by anomaly_detector)
        NaN    — month has zero median (no data); treated as 0.0

    Args:
        csv_path: Absolute or relative path to solar_report_6months.csv.
                  Expected columns: date, generation, feedin, gridConsumption.

    Returns:
        Annotated DataFrame with two extra columns:
            expected_generation  (float, kWh)
            efficiency_ratio     (float)
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])

    df["month"] = df["date"].dt.month

    monthly_median = df.groupby("month")["generation"].median()

    df["expected_generation"] = df["month"].map(monthly_median)

    df["efficiency_ratio"] = df["generation"] / df["expected_generation"].replace(0, float("nan"))
    df["efficiency_ratio"] = df["efficiency_ratio"].fillna(0.0)

    return df
