import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forecast.logging_store import PredictionLog


class LoggingTest(unittest.TestCase):
    def test_round_trip_and_limit(self):
        with tempfile.TemporaryDirectory() as root:
            log = PredictionLog(Path(root) / "test.sqlite")
            self.assertEqual(log.read(), [])
            log.append({"country": "all", "predicted_revenue": 10}, 1.5)
            log.append({"country": "france", "predicted_revenue": 2}, 2.5)
            event, = log.read(limit=1)
            self.assertEqual(event["country"], "france")
            self.assertEqual(event["latency_ms"], 2.5)
            self.assertIn("logged_at", event)
            self.assertEqual(len(log.read()), 2)

    def test_concurrent_writes_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            log = PredictionLog(Path(root) / "test.sqlite")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: log.append({"request": i}, 1), range(16)))
            self.assertEqual({row["request"] for row in log.read()}, set(range(16)))
