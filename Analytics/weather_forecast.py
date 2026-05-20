"""Daily weather forecast via Open-Meteo (free, keyless)."""

import requests

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes → human-readable description.
_WMO_CODES: dict[int, str] = {
    0:  "clear sky",
    1:  "mainly clear",
    2:  "partly cloudy",
    3:  "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_forecast(lat: float = 33.6007, lon: float = 73.0679, days: int = 7) -> list[dict]:
    """N-day forecast as a list of {date, rain_mm, cloud_cover_pct, description}.
    Default lat/lon is Rawalpindi; days is clamped to [1, 16]."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,cloud_cover_mean,weather_code",
        "timezone": "auto",
        "forecast_days": max(1, min(days, 16)),
    }

    response = requests.get(_OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    daily = response.json().get("daily", {})

    dates  = daily.get("time", [])
    rains  = daily.get("precipitation_sum", [])
    clouds = daily.get("cloud_cover_mean", [])
    codes  = daily.get("weather_code", [])

    result = []
    for i, date_str in enumerate(dates):
        result.append({
            "date":            date_str,
            "rain_mm":         round(float(rains[i] or 0.0), 2),
            "cloud_cover_pct": round(float(clouds[i] or 0.0), 1),
            "description":     _WMO_CODES.get(int(codes[i] or 0), f"WMO code {codes[i]}"),
        })
    return result
