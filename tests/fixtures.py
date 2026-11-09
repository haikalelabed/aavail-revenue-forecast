import numpy as np
import pandas as pd


def daily_fixture():
    dates = pd.date_range("2021-01-01", periods=430, name="date")
    step = np.arange(len(dates))
    return pd.DataFrame({"united kingdom": 100 + step / 3 + 9 * np.sin(step / 12),
                         "france": 30 + 4 * np.cos(step / 7)}, index=dates).assign(
                             all=lambda frame: frame.sum(axis=1))


def baseline_bundle(daily):
    from forecast.model import BASELINE

    return {"model_id": "test-bundle", "daily": daily,
            "estimators": {country: (BASELINE, None) for country in daily.columns}}
