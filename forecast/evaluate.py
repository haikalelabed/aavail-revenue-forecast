"""Generate EDA, holdout comparisons and a production replay via the API."""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from forecast.api import create_app
from forecast.ingest import ALL, ingest
from forecast.model import BASELINE, DAY, fit_models, save_bundle
from forecast.monitor import report


def eda(daily, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot(daily.index, daily[ALL].rolling(30).sum(), color="#047c77")
    axes[0].set(title="Global trailing 30-day revenue", ylabel="Revenue (source units)")
    axes[0].tick_params(axis="x", rotation=30)
    totals = daily.drop(columns=[ALL]).sum().nlargest(8).sort_values()
    axes[1].barh(totals.index.str.title(), totals, color="#326889")
    axes[1].set(title="Eight largest markets by training revenue", xlabel="Revenue (source units)")
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "eda.png", dpi=150)
    plt.close(fig)


def comparison(scores, output):
    world = scores[scores["country"] == ALL].copy()
    names = {BASELINE: "Trailing 30 days", "ridge": "Ridge", "random_forest": "Random forest",
             "gradient_boosting": "Gradient boosting"}
    labels = [names[row.model] + (" *" if row.deployed else "") for row in world.itertuples()]
    colors = ["#c6712a" if name == BASELINE else "#047c77" for name in world["model"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.3), layout="constrained")
    bars = ax.barh(labels, world["mae"], color=colors)
    ax.bar_label(bars, fmt="{:,.0f}", padding=5)
    ax.margins(x=.2)
    ax.invert_yaxis()
    ax.set(title="Global revenue forecast | Held-out MAE",
           xlabel="Mean absolute error (source units) · Lower is better\n"
                  "* Selected on earlier validation")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(output / "holdout_comparison.png", dpi=150)
    plt.close(fig)


def evaluate(train_source, production_source, output):
    output.mkdir(parents=True, exist_ok=True)
    daily, train_audit = ingest(train_source)
    observed, observed_audit = ingest(production_source)
    bundle, scores = fit_models(daily)
    scores.to_csv(output / "holdout_metrics.csv", index=False)
    (output / "ingestion_audit.json").write_text(
        json.dumps({"training": train_audit, "production": observed_audit}, indent=2) + "\n")
    eda(daily, output)
    comparison(scores, output)
    country = daily.drop(columns=[ALL]).sum().idxmax()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model, log, feed = root / "model.joblib", root / "predictions.sqlite", root / "observed.csv"
        save_bundle(bundle, model)
        observed.to_csv(feed)
        app = create_app(model_path=model, log_path=log, observed_path=feed)
        with app.test_client() as client:
            for origin in observed.index.insert(0, observed.index[0] - DAY):
                for scope in (ALL, country):
                    result = client.get("/predict", query_string={
                        "country": scope, "as_of": str(origin.date())})
                    if result.status_code != 200:
                        raise RuntimeError(result.get_json())
        summary = report(log, observed, output)
    print(scores[scores["country"] == ALL].to_string(index=False))
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_source", type=Path)
    parser.add_argument("production_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evaluate(args.train_source, args.production_source, args.output)


if __name__ == "__main__":
    main()
