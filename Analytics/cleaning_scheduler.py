_RAIN_THRESHOLD_MM = 5.0     # rain sum over 5 days that triggers WAIT
_LOSS_THRESHOLD_PKR = 200.0  # daily PKR loss that triggers CLEAN NOW
_CLOUD_SUPPRESS_PCT = 60.0   # cloud cover above which loss display is suppressed


def recommend(forecast: list[dict], daily_loss: float) -> dict:
    """Decide the cleaning action based on weather forecast and economic loss.

    Decision logic (Section 2.7), evaluated in priority order:
        1. WAIT       — total rainfall in the next 5 days > 5 mm.
        2. CLEAN NOW  — daily_loss_pkr > 200 PKR (and no significant rain).
        3. MONITOR    — all other cases.

    Additionally, when today's cloud cover exceeds 60 %, `suppress_loss` is
    set to True so the webapp can hide the loss figure (overcast readings are
    unreliable).

    Args:
        forecast:    Output of weather_forecast.get_forecast — list of daily
                     dicts with keys rain_mm, cloud_cover_pct, description.
        daily_loss:  Daily PKR loss from pkr_loss_calculator.calculate_loss
                     (the "daily_loss_pkr" value).

    Returns:
        {
            "action":               str   "WAIT" | "CLEAN NOW" | "MONITOR",
            "reason":               str   human-readable explanation,
            "suppress_loss":        bool  True when today cloud_cover > 60%,
            "rain_5day_mm":         float sum of rain_mm over next 5 days,
            "today_cloud_pct":      float today's cloud_cover_pct,
        }
    """
    rain_5day = sum(day.get("rain_mm", 0.0) for day in forecast[:5])
    today_cloud = forecast[0].get("cloud_cover_pct", 0.0) if forecast else 0.0
    suppress_loss = today_cloud > _CLOUD_SUPPRESS_PCT

    if rain_5day > _RAIN_THRESHOLD_MM:
        action = "WAIT"
        reason = (
            f"Rain of {rain_5day:.1f} mm expected over the next 5 days — "
            "let rainfall clean the panels naturally before spending on manual cleaning."
        )
    elif daily_loss > _LOSS_THRESHOLD_PKR:
        action = "CLEAN NOW"
        reason = (
            f"Daily loss of PKR {daily_loss:.0f} exceeds the PKR {_LOSS_THRESHOLD_PKR:.0f} "
            "threshold and no significant rain is forecast — cleaning pays off immediately."
        )
    else:
        action = "MONITOR"
        reason = (
            f"Daily loss of PKR {daily_loss:.0f} is below the action threshold "
            "and rain is not imminent — continue monitoring panel performance."
        )

    return {
        "action":          action,
        "reason":          reason,
        "suppress_loss":   suppress_loss,
        "rain_5day_mm":    round(rain_5day, 2),
        "today_cloud_pct": round(today_cloud, 1),
    }
