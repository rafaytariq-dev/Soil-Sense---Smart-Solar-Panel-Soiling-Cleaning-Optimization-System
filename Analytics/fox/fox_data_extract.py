"""
FoxESS Cloud — 6-Month Solar Data Extraction to CSV
====================================================
Pulls daily production report data and optional granular history data
from the FoxESS Open API and saves everything to CSV files.

Usage:
    python fox_data_extract.py

Output:
    - solar_report_6months.csv   (daily totals: generation, feedin, grid consumption, etc.)
    - solar_history_6months.csv  (5-min interval data for detailed analysis)

API Reference:
    https://www.foxesscloud.com/public/i18n/en/OpenApiDocument.html
"""

import os
import sys
import csv
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory so we can import fox_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fox_config import api_request, get_device_sn

# ── Configuration ──
DAYS_TO_PULL = 180
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Report variables (daily aggregated data)
REPORT_VARIABLES = [
    "generation",           # Total generation (kWh)
    "feedin",               # Energy fed into grid (kWh)
    "gridConsumption",      # Energy consumed from grid (kWh)
    "chargeEnergyToTal",    # Battery charge energy (kWh)
    "dischargeEnergyToTal", # Battery discharge energy (kWh)
]

# History variables (5-min interval real-time data)
HISTORY_VARIABLES = [
    "pvPower",              # Total PV power (kW)
    "pv1Power",             # PV string 1 power (kW)
    "pv2Power",             # PV string 2 power (kW)
    "generationPower",      # Output power (kW)
    "feedinPower",          # Feed-in power to grid (kW)
    "gridConsumptionPower", # Grid consumption power (kW)
    "loadsPower",           # Load power (kW)
    "meterPower",           # Meter power (kW)
    "invTemperation",       # Inverter temperature (°C)
    "ambientTemperation",   # Ambient temperature (°C)
    "SoC",                  # Battery state of charge (%)
    "batChargePower",       # Battery charge power (kW)
    "batDischargePower",    # Battery discharge power (kW)
]


def fetch_report_data(sn: str, days: int = DAYS_TO_PULL) -> list:
    """
    Fetch daily report data for the last N days using the report/query endpoint.
    
    The report API works with dimension="month" to get daily data for a given month,
    or dimension="day" to get hourly data for a given day.
    
    We use dimension="month" and iterate over each month in our date range,
    then filter to only the days we need.
    
    Returns a list of dicts with date and energy values.
    """
    today = datetime.now()
    start_date = today - timedelta(days=days)
    
    print(f"\n{'='*60}")
    print(f"  FETCHING DAILY REPORT DATA")
    print(f"  Device SN: {sn}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')}")
    print(f"  ({days} days)")
    print(f"{'='*60}\n")

    # Determine which months we need to query
    months_to_query = set()
    current = start_date
    while current <= today:
        months_to_query.add((current.year, current.month))
        current += timedelta(days=1)
    
    months_to_query = sorted(months_to_query)
    all_daily_data = []

    for year, month in months_to_query:
        print(f"  [REPORT] Querying {year}-{month:02d}...", end=" ", flush=True)
        
        try:
            resp = api_request('post', '/op/v0/device/report/query', {
                "sn": sn,
                "year": year,
                "month": month,
                "dimension": "month",
                "variables": REPORT_VARIABLES
            })
            
            result = resp.get('result', [])
            
            if not result:
                print("(no data)")
                continue
            
            # Parse the response - each variable has an array of daily values
            # The result is a list of dicts, one per variable
            # Each dict has: { "variable": "generation", "unit": "kWh", "values": [{...}, ...] }
            
            # Determine how many days are in this month
            import calendar
            days_in_month = calendar.monthrange(year, month)[1]
            
            # Build daily records
            for day_idx in range(days_in_month):
                day_num = day_idx + 1
                record_date = datetime(year, month, day_num)
                
                # Skip dates outside our range
                if record_date.date() < start_date.date() or record_date.date() > today.date():
                    continue
                
                row = {"date": record_date.strftime("%Y-%m-%d")}
                
                for var_data in result:
                    var_name = var_data.get("variable", "")
                    values = var_data.get("values", [])
                    
                    if day_idx < len(values):
                        raw = values[day_idx]
                        # Values can be raw floats/ints OR dicts like {"value": x}
                        if isinstance(raw, dict):
                            val = raw.get("value", 0)
                        else:
                            val = raw
                        row[var_name] = val if val is not None else 0
                    else:
                        row[var_name] = 0
                
                all_daily_data.append(row)
            
            print(f"✓ ({len(result)} variables)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    print(f"\n  Total daily records fetched: {len(all_daily_data)}")
    return all_daily_data


def fetch_history_data(sn: str, days: int = DAYS_TO_PULL) -> list:
    """
    Fetch granular (5-minute interval) history data for the last N days.
    
    Uses /op/v0/device/history/query which supports a 24h window per call.
    We loop day-by-day.
    
    WARNING: This makes one API call per day, so 60 days = 60 calls.
    FoxESS limit is 1440 calls/day per inverter.
    
    Returns a list of dicts with timestamp and variable values.
    """
    today = datetime.now()
    start_date = today - timedelta(days=days)
    
    print(f"\n{'='*60}")
    print(f"  FETCHING GRANULAR HISTORY DATA (5-min intervals)")
    print(f"  Device SN: {sn}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} → {today.strftime('%Y-%m-%d')}")
    print(f"  ({days} days × ~288 data points/day)")
    print(f"{'='*60}\n")

    all_history = []
    current_date = start_date

    for day_num in range(days):
        day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        begin_ts = int(day_start.timestamp() * 1000)
        end_ts = int(day_end.timestamp() * 1000)
        
        date_str = day_start.strftime('%Y-%m-%d')
        print(f"  [HISTORY] Day {day_num+1}/{days}: {date_str}...", end=" ", flush=True)
        
        try:
            resp = api_request('post', '/op/v0/device/history/query', {
                "sn": sn,
                "variables": HISTORY_VARIABLES,
                "begin": begin_ts,
                "end": end_ts
            })
            
            result = resp.get('result', [])
            
            if not result:
                print("(no data)")
                current_date += timedelta(days=1)
                continue
            
            # Parse history response
            # Actual format: result = [{"deviceSN": "...", "datas": [{"unit": "kW", "variable": "pvPower", "data": [{"time": "...", "value": 0.0}, ...]}, ...]}]
            # The datas array contains one entry per variable
            
            device_data = result[0] if isinstance(result, list) and len(result) > 0 else result
            datas = device_data.get("datas", [])
            
            if not datas:
                print("(empty)")
                current_date += timedelta(days=1)
                continue
            
            # Find the variable with the most data points to determine timestamps
            max_points = 0
            time_points = []
            
            for var_data in datas:
                data_points = var_data.get("data", [])
                if len(data_points) > max_points:
                    max_points = len(data_points)
                    time_points = [p.get("time", "") for p in data_points]
            
            if max_points == 0:
                print("(empty)")
                current_date += timedelta(days=1)
                continue
            
            # Build records for each timestamp
            for i in range(max_points):
                row = {
                    "date": date_str,
                    "timestamp": time_points[i] if i < len(time_points) else ""
                }
                
                for var_data in datas:
                    var_name = var_data.get("variable", "")
                    data_points = var_data.get("data", [])
                    
                    if i < len(data_points):
                        val = data_points[i].get("value", 0)
                        row[var_name] = val if val is not None else 0
                    else:
                        row[var_name] = 0
                
                all_history.append(row)
            
            print(f"ok ({max_points} data points)")
            
        except Exception as e:
            print(f"ERROR: {e}")
        
        current_date += timedelta(days=1)

    print(f"\n  Total history records fetched: {len(all_history)}")
    return all_history


def save_to_csv(data: list, filename: str) -> str:
    """Save a list of dicts to a CSV file. Returns the full file path."""
    if not data:
        print(f"  [WARN] No data to save for {filename}")
        return ""

    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Use the keys from the first record as columns
    fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    file_size = os.path.getsize(filepath) / 1024
    print(f"\n  ✓ Saved: {filepath}")
    print(f"    Rows: {len(data)} | Columns: {len(fieldnames)} | Size: {file_size:.1f} KB")
    return filepath


def main():
    print("\n" + "═"*60)
    print("  FoxESS Solar Data Extractor")
    print("  Pulling 6 months (180 days) of solar panel data")
    print("═"*60)
    
    # Step 1: Get device serial number
    print("\n[STEP 1] Detecting device...")
    try:
        sn = get_device_sn()
    except Exception as e:
        print(f"  ✗ Failed to get device info: {e}")
        print("  Check your API key in the .env file")
        sys.exit(1)
    
    # Step 2: Fetch daily report data (quick, few API calls)
    print("\n[STEP 2] Fetching daily report data...")
    report_data = fetch_report_data(sn, days=DAYS_TO_PULL)
    report_file = save_to_csv(report_data, "solar_report_6months.csv")
    
    # Step 3: Fetch granular history data (slower, one call per day)
    print("\n[STEP 3] Fetching granular history data...")
    print("  (This will take ~4-5 minutes due to API rate limits — 180 days of data)")
    history_data = fetch_history_data(sn, days=DAYS_TO_PULL)
    history_file = save_to_csv(history_data, "solar_history_6months.csv")
    
    # Summary
    print("\n" + "═"*60)
    print("  EXTRACTION COMPLETE!")
    print("═"*60)
    if report_file:
        print(f"  📊 Daily Report:  {report_file}")
    if history_file:
        print(f"  📈 History Data:  {history_file}")
    print(f"\n  Next step: Run 'python fox_visualize.py' to generate charts")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
