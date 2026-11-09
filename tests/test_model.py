import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast.model import BASELINE, HORIZON, features, fit_models, forecast, save_bundle
from tests.fixtures import baseline_bundle, daily_fixture


class ModelTest(unittest.TestCase):
    def test_features_ignore_future_revenue(self):
        daily = daily_fixture()["all"]
        date = daily.index[-31]
        expected = features(daily, date)
        daily.iloc[-30:] = 1e10
        self.assertEqual(features(daily, date), expected)

    def test_bundle_round_trip_and_forecast_window(self):
        daily = daily_fixture()
        bundle = baseline_bundle(daily)
        origin = daily.index[-1]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.joblib"
            save_bundle(bundle, path)
            result = forecast(joblib.load(path), daily, "all", origin)
        self.assertAlmostEqual(result["predicted_revenue"], daily["all"].tail(30).sum(), places=3)
        self.assertEqual(result["forecast_start"], str((origin + pd.Timedelta(days=1)).date()))
        self.assertEqual(result["forecast_end"], str((origin + pd.Timedelta(days=HORIZON)).date()))

    def test_candidate_comparison_returns_finite_metrics(self):
        daily = daily_fixture()
        bundle, scores = fit_models(daily)
        self.assertEqual(set(bundle["estimators"]), set(daily.columns))
        self.assertEqual(set(scores["model"]), {BASELINE, "ridge", "random_forest", "gradient_boosting"})
        self.assertTrue(np.isfinite(scores[["mae", "rmse", "bias", "validation_mae"]]).all().all())
        self.assertTrue((scores.groupby("country")["deployed"].sum() == 1).all())
