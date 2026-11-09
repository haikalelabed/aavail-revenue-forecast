"""Read course invoice JSON into a daily revenue table."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALL = "all"


def ingest(directory):
    """Return daily revenue by country plus a separately calculated global total."""
    files = sorted(Path(directory).glob("invoices-*.json"))
    if not files:
        raise ValueError(f"No invoices-*.json files in {directory}")
    aggregates = []
    audit = {"files": len(files), "rows_read": 0, "rows_rejected": 0,
             "missing_country_rows": 0, "negative_price_rows": 0}
    for path in files:
        rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        price_column = "price" if "price" in rows else "total_price"
        required = {"country", "year", "month", "day", price_column}
        if not required.issubset(rows.columns):
            raise ValueError(f"Unexpected invoice schema in {path.name}")
        country = rows["country"].astype("string").str.strip().str.casefold()
        missing_country = country.isna() | country.isin(["", "unspecified"])
        country = country.mask(missing_country, "unknown")
        if (country == ALL).any():
            raise ValueError("'all' is reserved for the global total")
        dates = pd.to_datetime(rows[["year", "month", "day"]], errors="coerce")
        price = pd.to_numeric(rows[price_column], errors="coerce")
        valid = dates.notna() & np.isfinite(price)
        audit["rows_read"] += len(rows)
        audit["rows_rejected"] += int((~valid).sum())
        audit["missing_country_rows"] += int((valid & missing_country).sum())
        audit["negative_price_rows"] += int((valid & (price < 0)).sum())
        clean = pd.DataFrame({"date": dates, "country": country, "revenue": price})
        aggregates.append(clean.loc[valid].groupby(["date", "country"])["revenue"].sum())
    combined = pd.concat(aggregates)
    if combined.empty:
        raise ValueError(f"No valid invoice rows in {directory}")
    daily = combined.groupby(level=["date", "country"]).sum().unstack(fill_value=0)
    daily = daily.asfreq("D", fill_value=0).astype(float)
    daily[ALL] = daily.sum(axis=1)
    daily.index.name = "date"
    audit.update(first_day=str(daily.index[0].date()), last_day=str(daily.index[-1].date()),
                 country_labels=len(daily.columns) - 1,
                 zero_revenue_days=int((daily[ALL] == 0).sum()))
    return daily, audit


def read_daily(path):
    return pd.read_csv(path, index_col="date", parse_dates=["date"], float_precision="round_trip")


def combine_history(training, observed):
    """Append only newer observations and preserve training history."""
    newer = observed.loc[observed.index > training.index[-1]]
    if newer.empty:
        return training
    if newer.index[0] - training.index[-1] != pd.Timedelta(days=1):
        raise ValueError("Observed data must continue the training history without a gap")
    return pd.concat([training, newer]).fillna(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    daily, audit = ingest(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
