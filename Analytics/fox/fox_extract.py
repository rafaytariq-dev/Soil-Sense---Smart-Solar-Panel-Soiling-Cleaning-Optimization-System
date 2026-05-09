"""
FoxESS Cloud — Device Discovery & Diagnostics
================================================
Quick script to verify API connection and discover your device info.
Run this first to confirm your API key works and get your device SN.

Usage:
    python fox_extract.py
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory so we can import fox_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fox_config import api_request, get_device_list, get_device_sn, API_KEY


def main():
    print("\n" + "═"*60)
    print("  FoxESS Cloud — Device Discovery & Diagnostics")
    print("═"*60)

    # 1. Verify API Key
    print(f"\n[1] API Key: {API_KEY[:8]}...{API_KEY[-4:]}" if len(API_KEY) > 12 else f"\n[1] API Key: {API_KEY}")
    
    if not API_KEY:
        print("  ✗ No API key found! Check your .env file.")
        sys.exit(1)

    # 2. Get Device List
    print("\n[2] Fetching device list...")
    try:
        devices = get_device_list()
        print(f"  ✓ Found {len(devices)} device(s)\n")
        
        for i, device in enumerate(devices):
            print(f"  ── Device {i+1} ──")
            print(f"  Serial Number:  {device.get('deviceSN', 'N/A')}")
            print(f"  Device Type:    {device.get('deviceType', 'N/A')}")
            print(f"  Module SN:      {device.get('moduleSN', 'N/A')}")
            print(f"  Plant Name:     {device.get('plantName', 'N/A')}")
            print(f"  Status:         {_status_text(device.get('status', 0))}")
            print(f"  Power (now):    {device.get('power', 0)} kW")
            print(f"  Generation:     {device.get('generationPower', 0)} kWh (today)")
            print()
    
    except Exception as e:
        print(f"  ✗ Error fetching devices: {e}")
        sys.exit(1)

    # 3. Get device generation summary
    if devices:
        sn = devices[0].get('deviceSN', '')
        print(f"[3] Fetching generation summary for {sn}...")
        try:
            resp = api_request('get', '/op/v0/device/generation', {'sn': sn})
            gen = resp.get('result', {})
            print(f"  Today:     {gen.get('today', 0)} kWh")
            print(f"  Month:     {gen.get('month', 0)} kWh")
            print(f"  Year:      {gen.get('year', 0)} kWh")
            print(f"  Cumulative: {gen.get('cumulative', 0)} kWh")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "═"*60)
    print("  ✓ Diagnostics complete!")
    print("  Next: Run 'python fox_data_extract.py' to pull 6 months of data")
    print("═"*60 + "\n")


def _status_text(status: int) -> str:
    """Convert device status code to text."""
    return {1: "🟢 Online", 2: "🔴 Fault", 3: "⚫ Offline"}.get(status, f"Unknown ({status})")


if __name__ == "__main__":
    main()