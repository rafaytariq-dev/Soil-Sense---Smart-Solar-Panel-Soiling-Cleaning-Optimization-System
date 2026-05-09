"""
FoxESS Cloud Open API - Configuration & Authentication Helper
==============================================================
Handles authentication via API key (token-based MD5 signature)
and rate-limited requests.

Based on: https://www.foxesscloud.com/public/i18n/en/OpenApiDocument.html

NOTE: The API key is your 'token' from the FoxESS Cloud platform.
      Go to: User Profile > API Management > generate your api-key.
      The key should be a ~36 character UUID-like string.
"""

import os
import sys
import time
import hashlib
import json
import requests
from dotenv import load_dotenv

# ── Load .env from the same directory as this script ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

# ── Configuration ──
API_KEY = os.getenv("FOXESS_API_KEY", "")
USERNAME = os.getenv("FOXESS_USERNAME", "")
PASSWORD = os.getenv("FOXESS_PASSWORD", "")
DOMAIN = "https://www.foxesscloud.com"
LANG = "en"

# Rate limiting: FoxESS allows 1 query/sec per endpoint, 1440 total calls/day
_last_request_time = 0
MIN_REQUEST_INTERVAL = 1.1  # seconds between requests

# ── Session token cache (for username/password auth) ──
_session_token = None
_session_token_expiry = 0


def _get_session_token() -> str:
    """
    Authenticate via username/password to get a session token.
    This is used when the API key doesn't work or isn't provided.
    The foxesscloud.com login endpoint returns a token valid for ~2 hours.
    """
    global _session_token, _session_token_expiry

    # Return cached token if still valid
    if _session_token and time.time() < _session_token_expiry:
        return _session_token

    print("  [AUTH] Logging in with username/password...", end=" ", flush=True)

    login_url = f"{DOMAIN}/c/v0/user/login"
    login_payload = {
        "user": USERNAME,
        "password": hashlib.md5(PASSWORD.encode('UTF-8')).hexdigest()
    }
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'lang': LANG,
    }

    resp = requests.post(login_url, json=login_payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get('errno', 0) != 0:
        raise Exception(f"Login failed: {data.get('msg', 'Unknown error')}")

    _session_token = data.get('result', {}).get('token', '')
    _session_token_expiry = time.time() + 7000  # Token valid for ~2 hours, refresh early

    if not _session_token:
        raise Exception("Login succeeded but no token returned")

    print("OK")
    return _session_token


def _get_auth_token() -> str:
    """
    Get the best available authentication token.
    Prefers API key, falls back to username/password session token.
    """
    if API_KEY and len(API_KEY) >= 36:
        return API_KEY
    else:
        if USERNAME and PASSWORD:
            return _get_session_token()
        else:
            raise Exception(
                "No valid API key or username/password found!\n"
                "Set FOXESS_API_KEY (36+ chars) or FOXESS_USERNAME + FOXESS_PASSWORD in .env"
            )


def _generate_signature(path: str, token: str) -> dict:
    """
    Generate the authentication headers required by the FoxESS Open API.
    Signature = MD5( path + "\\r\\n" + token + "\\r\\n" + timestamp )
    """
    timestamp = str(round(time.time() * 1000))
    # IMPORTANT: FoxESS expects literal \r\n (backslash-r-backslash-n), NOT actual CR+LF
    signature_string = fr'{path}\r\n{token}\r\n{timestamp}'
    signature = hashlib.md5(signature_string.encode('UTF-8')).hexdigest()

    return {
        'token': token,
        'lang': LANG,
        'timestamp': timestamp,
        'signature': signature,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }


def api_request(method: str, path: str, params: dict = None) -> dict:
    """
    Make a rate-limited request to the FoxESS Open API.

    Args:
        method: 'get' or 'post'
        path:   API path, e.g. '/op/v0/device/list'
        params: Query params (GET) or JSON body (POST)

    Returns:
        Parsed JSON response as a dict.
    """
    global _last_request_time

    # Rate limiting
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    token = _get_auth_token()
    url = DOMAIN + path
    headers = _generate_signature(path, token)

    if method.lower() == 'get':
        response = requests.get(url, params=params, headers=headers, timeout=30)
    elif method.lower() == 'post':
        response = requests.post(url, json=params, headers=headers, timeout=30)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    _last_request_time = time.time()

    # Handle token expiry — if 401, try re-authenticating once
    if response.status_code == 401:
        global _session_token, _session_token_expiry
        _session_token = None
        _session_token_expiry = 0
        token = _get_auth_token()
        headers = _generate_signature(path, token)
        if method.lower() == 'get':
            response = requests.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.post(url, json=params, headers=headers, timeout=30)

    response.raise_for_status()
    data = response.json()

    # FoxESS returns errno != 0 on API-level errors
    if data.get('errno', 0) != 0:
        raise Exception(
            f"FoxESS API error (errno={data.get('errno')}): {data.get('msg', 'Unknown error')}"
        )

    return data


def get_device_list() -> list:
    """Fetch all devices (inverters) under this account."""
    resp = api_request('post', '/op/v0/device/list', {
        'currentPage': 1,
        'pageSize': 100
    })
    devices = resp.get('result', {}).get('data', [])
    return devices


def get_device_sn() -> str:
    """Get the serial number of the first device on this account."""
    devices = get_device_list()
    if not devices:
        raise Exception("No devices found on this account!")
    sn = devices[0].get('deviceSN', '')
    device_type = devices[0].get('deviceType', 'Unknown')
    status = {1: 'Online', 2: 'Fault', 3: 'Offline'}.get(devices[0].get('status', 0), 'Unknown')
    print(f"  [INFO] Found device: {device_type} | SN: {sn} | Status: {status}")
    return sn
