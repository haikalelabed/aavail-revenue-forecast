"""Match logged forecasts to mature outcomes and evaluate business-scale error."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from forecast.ingest import read_daily
from forecast.logging_store import PredictionLog
from forecast.model import metrics


def score_events(events, observed):
    latest = {}
    for event in events:  # The log returns newest first.
        key = (event["country"], event["as_of"], event["model_id"])
        latest.setdefault(key, event)
    scored = []
    for event in latest.values():
        country = event["country"]
        start, end = pd.Timestamp(event["forecast_start"]), pd.Timestamp(event["forecast_end"])
        days = pd.date_range(start, end)
        if country not in observed or not days.isin(observed.index).all():
            continue
        actual = float(observed.loc[days, country].sum())
        scored.append({**event, "actual_revenue": actual,
                       "absolute_error": abs(event["predicted_revenue"] - actual)})
    return pd.DataFrame(scored).sort_values("as_of") if scored else pd.DataFrame()


def report(log_path, observed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scored = score_events(PredictionLog(log_path).read(), observed)
    if scored.empty:
        raise ValueError("No forecasts with fully observed outcome windows")
    scored.to_csv(output / "monitor_daily.csv", index=False)
    summaries = []
    groups = scored.groupby(["country", "model_id"])
    for (country, model_id), group in groups:
        actual = group["actual_revenue"]
        model = metrics(actual, group["predicted_revenue"])
        baseline = metrics(actual, group["baseline_revenue"])
        denominator = actual.abs().sum()
        summaries.append({"country": country, "model_id": model_id, "model": group["model"].iloc[0],
                          "scored_forecasts": len(group), **model, "baseline_mae": baseline["mae"],
                          "wape": model["mae"] * len(group) / denominator if denominator else None,
                          "baseline_wape": baseline["mae"] * len(group) / denominator if denominator else None,
                          "p95_latency_ms": group["latency_ms"].quantile(.95),
                          "baseline_beats_model": baseline["mae"] < model["mae"]})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "monitor_summary.csv", index=False)

    world = scored[scored["country"] == "all"]
    if not world.empty:
        fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
        dates = pd.to_datetime(world["as_of"])
        ax.plot(dates, world["actual_revenue"], color="#14334b", linewidth=2.4, label="Actual")
        ax.plot(dates, world["predicted_revenue"], color="#047c77", label="Deployed model")
        if not world["predicted_revenue"].equals(world["baseline_revenue"]):
            ax.plot(dates, world["baseline_revenue"], color="#c6712a", linestyle="--", label="Baseline")
        ax.set(title="Production replay | Global 30-day revenue", xlabel="Forecast origin",
               ylabel="Revenue (source units)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output / "production_replay.png", dpi=150)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("observed_daily")
    parser.add_argument("output")
    args = parser.parse_args()
    print(report(args.log, read_daily(args.observed_daily), args.output).to_string(index=False))


if __name__ == "__main__":
    main()
