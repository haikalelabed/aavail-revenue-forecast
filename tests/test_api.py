import tempfile
import unittest
from pathlib import Path

from forecast.api import create_app
from forecast.model import save_bundle
from tests.fixtures import baseline_bundle, daily_fixture


class APITest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.model = root / "model.joblib"
        self.log = root / "predictions.sqlite"
        self.training = root / "train.csv"
        self.daily = daily_fixture()
        self.daily.to_csv(self.training)
        self.date = self.daily.index[-1].date().isoformat()
        save_bundle(baseline_bundle(self.daily), self.model)
        self.client = create_app(model_path=self.model, log_path=self.log,
                                 training_path=self.training).test_client()

    def test_specific_country_and_all(self):
        response = self.client.post("/predict", json={"country": "United Kingdom", "as_of": self.date})
        self.assertEqual(response.status_code, 200)
        country = response.get_json()
        response = self.client.get("/predict", query_string={"country": "all", "as_of": self.date})
        self.assertEqual(response.status_code, 200)
        total = response.get_json()
        self.assertEqual(country["country"], "united kingdom")
        self.assertEqual(total["country"], "all")
        for scope, result in (("united kingdom", country), ("all", total)):
            self.assertAlmostEqual(result["predicted_revenue"], self.daily[scope].tail(30).sum(), places=3)
        self.assertEqual(len(self.client.get("/logfile").get_json()["events"]), 2)

    def test_bad_inputs_do_not_log_predictions(self):
        for payload, status in (
            ({"country": "atlantis", "as_of": self.date}, 404),
            ({"country": 1, "as_of": self.date}, 400),
            ({"as_of": "yesterday"}, 400),
            ({"as_of": "2021-01-01"}, 422),
            ({"as_of": "2030-01-01"}, 422),
            (["wrong shape"], 400),
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, status)
        self.assertFalse(self.log.exists())

    def test_train_and_health(self):
        self.model.unlink()
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.client.get("/predict", query_string={"as_of": self.date}).status_code, 503)
        response = self.client.post("/train")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["countries"], 2)
        self.assertEqual(self.client.get("/health").status_code, 200)
