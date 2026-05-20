import math


def calculate_loss(
    watt_loss: float,
    system_kw: float = 10,
    tariff: float = 65,
    cleaning_cost: float = 800,
) -> dict:
    """Translate a predicted watt-drop into PKR loss figures.

    Formula (Section 2.7):
        daily_loss_pkr = (watt_loss / 1000) * 6 * tariff

    The 6-hour factor represents Rawalpindi's average daily peak-sun hours.
    `watt_loss` is the drop in panel output in Watts (W); it is converted to
    kW before the formula is applied.

    Args:
        watt_loss:     Predicted power drop in Watts (from dual_head_predictor).
        system_kw:     Installed system capacity in kW (default 10).
        tariff:        Electricity price in PKR per kWh (default 65).
        cleaning_cost: Cost of one cleaning in PKR (default 800).
                       Pass 0 for DIY cleaning — days_to_breakeven becomes 0.

    Returns:
        {
            "daily_loss_pkr":    float  — estimated PKR lost per day due to soiling,
            "monthly_loss_pkr":  float  — daily × 30,
            "loss_pct":          float  — watt_loss as % of system peak output,
            "days_to_breakeven": int    — cleaning_cost / daily_loss, rounded up;
                                          0 if cleaning_cost == 0 (DIY);
                                         -1 if daily_loss == 0 (nothing to recover),
            "clean_recommended": bool   — True when daily_loss_pkr >= 200.
        }
    """
    watt_loss_kw = max(watt_loss, 0.0) / 1000.0
    daily_loss_pkr = watt_loss_kw * 6.0 * tariff
    monthly_loss_pkr = daily_loss_pkr * 30.0
    loss_pct = round((watt_loss_kw / system_kw) * 100.0, 2) if system_kw > 0 else 0.0

    if cleaning_cost == 0:
        days_to_breakeven = 0
    elif daily_loss_pkr == 0:
        days_to_breakeven = -1
    else:
        days_to_breakeven = math.ceil(cleaning_cost / daily_loss_pkr)

    return {
        "daily_loss_pkr":    round(daily_loss_pkr, 2),
        "monthly_loss_pkr":  round(monthly_loss_pkr, 2),
        "loss_pct":          loss_pct,
        "days_to_breakeven": days_to_breakeven,
        "clean_recommended": daily_loss_pkr >= 200.0,
    }
