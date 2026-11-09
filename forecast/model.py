"""Fit, evaluate and persist direct 30-day revenue forecasts."""

import argparse
import hashlib
import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 30
DAY = pd.Timedelta(days=1)
WINDOWS = (7, 14, 30, 60, 90)
BASELINE = "trailing_30_days"
CANDIDATES = {
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "random_forest": RandomForestRegressor(
        n_estimators=80, max_depth=5, min_samples_leaf=3, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=80, max_depth=2, min_samples_leaf=4, random_state=42),
}


def features(series, origin):
    past = series.loc[:origin]
    if origin not in past.index or len(past) < max(WINDOWS):
        raise ValueError("Forecast date needs 90 days of history ending on that date")
    angle = 2 * math.pi * (origin.dayofyear - 1) / 365.25
    return [float(past.iloc[-n:].sum()) for n in WINDOWS] + [math.sin(angle), math.cos(angle)]


def predict_rows(estimator, x):
    return x[:, WINDOWS.index(HORIZON)] if estimator is None else estimator.predict(x)


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error))}


def fit_models(daily):
    """Choose per-scope models on validation, then score once on a later holdout."""
    if len(daily) < 400:
        raise ValueError("Training requires at least 400 days of invoice history")
    last = daily.index[-1]
    validation_start, test_start = last - 180 * DAY, last - 90 * DAY
    origins = daily.index[max(WINDOWS) - 1: -HORIZON: 7]
    label_ends = origins + HORIZON * DAY
    train = label_ends < validation_start
    validation = (origins >= validation_start) & (label_ends < test_start)
    pretest = label_ends < test_start
    test = origins >= test_start
    estimators, scores = {}, []
    for country in daily.columns:
        series = daily[country]
        x = np.array([features(series, origin) for origin in origins])
        y = np.array([series.loc[origin + DAY:origin + HORIZON * DAY].sum()
                      for origin in origins])
        fitted = {BASELINE: None}
        # A zero-history scope can still use the simple baseline.
        if np.any(y[train]):
            fitted.update({name: clone(model).fit(x[train], y[train])
                           for name, model in CANDIDATES.items()})
        validation_mae = {name: metrics(y[validation], predict_rows(model, x[validation]))["mae"]
                          for name, model in fitted.items()}
        winner = min(validation_mae, key=validation_mae.get)
        for name in fitted:
            model = None if name == BASELINE else clone(CANDIDATES[name]).fit(x[pretest], y[pretest])
            scores.append({"country": country, "model": name, "deployed": name == winner,
                           "validation_mae": validation_mae[name], "holdout_origins": int(test.sum()),
                           **metrics(y[test], predict_rows(model, x[test]))})
        deployed = None if winner == BASELINE else clone(CANDIDATES[winner]).fit(x, y)
        estimators[country] = (winner, deployed)
    identity = hashlib.sha256(pd.util.hash_pandas_object(daily, index=True).values.tobytes())
    identity.update(repr((tuple(daily.columns), WINDOWS, HORIZON, CANDIDATES)).encode())
    bundle = {"model_id": identity.hexdigest()[:16], "daily": daily, "estimators": estimators}
    return bundle, pd.DataFrame(scores)


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        joblib.dump(bundle, pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def train_file(daily_path, model_path):
    from forecast.ingest import read_daily

    bundle, scores = fit_models(read_daily(daily_path))
    save_bundle(bundle, model_path)
    return bundle, scores


def forecast(bundle, daily, country, origin):
    if country not in bundle["estimators"]:
        raise KeyError(country)
    if origin < bundle["daily"].index[-1]:
        raise ValueError("Forecast date precedes this model's training cutoff")
    name, model = bundle["estimators"][country]
    x = np.asarray([features(daily[country], origin)])
    return {
        "country": country,
        "as_of": str(origin.date()),
        "forecast_start": str((origin + DAY).date()),
        "forecast_end": str((origin + HORIZON * DAY).date()),
        "predicted_revenue": round(float(predict_rows(model, x)[0]), 4),
        "baseline_revenue": round(float(predict_rows(None, x)[0]), 4),
        "model": name,
        "model_id": bundle["model_id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("daily")
    parser.add_argument("model")
    args = parser.parse_args()
    bundle, scores = train_file(args.daily, args.model)
    print(scores[scores["country"] == "all"].to_string(index=False))
    print(f"Saved model {bundle['model_id']} to {args.model}")


if __name__ == "__main__":
    main()
