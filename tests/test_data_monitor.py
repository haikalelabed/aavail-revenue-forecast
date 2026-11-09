import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast.ingest import ingest
from forecast.monitor import score_events


class IngestionAndMonitoringTest(unittest.TestCase):
    def test_ingestion_handles_aliases_missing_dates_and_negative_prices(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [{"country": " France ", "year": 2019, "month": 7, "day": day,
                     "total_price": price} for day, price in [(1, 12), (3, -2), (35, 5)]]
            (Path(root) / "invoices-2019-07.json").write_text(json.dumps(rows))
            data, audit = ingest(root)
        self.assertEqual(data["all"].tolist(), [12, 0, -2])
        self.assertEqual(data["france"].tolist(), [12, 0, -2])
        self.assertEqual(audit["rows_rejected"], 1)
        self.assertEqual(audit["negative_price_rows"], 1)

    def test_monitor_scores_only_mature_and_unique_predictions(self):
        observed = pd.DataFrame({"all": 2}, index=pd.date_range("2019-08-01", periods=30))
        event = {"country": "all", "as_of": "2019-07-31", "forecast_start": "2019-08-01",
                 "forecast_end": "2019-08-30", "predicted_revenue": 55, "model_id": "test"}
        immature = dict(event, as_of="2019-08-01", forecast_start="2019-08-02",
                        forecast_end="2019-08-31")
        scored = score_events([event, event, immature], observed)
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["actual_revenue"], 60)
        self.assertEqual(scored.iloc[0]["absolute_error"], 5)
