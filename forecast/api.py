"""Flask routes for training, prediction and inspection of forecast logs."""

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from forecast.ingest import ALL, combine_history, read_daily
from forecast.logging_store import PredictionLog
from forecast.model import forecast, train_file


def create_app(model_path=None, log_path=None, training_path=None, observed_path=None):
    app = Flask(__name__)
    model_path = Path(model_path or os.getenv("MODEL_PATH", "runtime/model.joblib"))
    log = PredictionLog(log_path or os.getenv("LOG_PATH", "runtime/predictions.sqlite"))
    training_path = training_path or os.getenv("TRAINING_DAILY_PATH")
    observed_path = observed_path or os.getenv("OBSERVED_DAILY_PATH")
    train_lock = Lock()

    @lru_cache(maxsize=1)
    def load_current_bundle(version):
        # Atomic model replacement changes the inode and invalidates this cache.
        return joblib.load(model_path)

    @app.get("/health")
    def health():
        ready = model_path.is_file()
        return jsonify(status="ready" if ready else "model_missing"), 200 if ready else 503

    @app.post("/train")
    def train():
        if not training_path:
            return jsonify(error="Training is not configured on this service"), 503
        if not train_lock.acquire(blocking=False):
            return jsonify(error="Training already in progress"), 409
        try:
            bundle, _ = train_file(training_path, model_path)
        finally:
            train_lock.release()
        return jsonify(model_id=bundle["model_id"], countries=len(bundle["estimators"]) - 1)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        started = perf_counter()
        payload = request.args if request.method == "GET" else request.get_json(silent=True)
        if not hasattr(payload, "get"):
            return jsonify(error="Provide a JSON object"), 400
        country = payload.get("country", ALL)
        if not isinstance(country, str) or not country.strip():
            return jsonify(error="Country must be a nonempty string"), 400
        country = country.strip().casefold()
        raw_date = payload.get("as_of", payload.get("date"))
        try:
            origin = pd.Timestamp(date.fromisoformat(raw_date))
            if raw_date != origin.date().isoformat():
                raise ValueError
        except (ValueError, TypeError):
            return jsonify(error="Provide as_of in YYYY-MM-DD format"), 400
        try:
            version = model_path.stat()
        except FileNotFoundError:
            return jsonify(error="Train the model first"), 503
        bundle = load_current_bundle((version.st_ino, version.st_mtime_ns))
        history = bundle["daily"]
        try:
            if observed_path:
                history = combine_history(history, read_daily(observed_path))
            result = forecast(bundle, history, country, origin)
        except KeyError:
            return jsonify(error=f"Unknown country: {country}"), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 422
        log.append(result, (perf_counter() - started) * 1000)
        return jsonify(result)

    @app.get("/logfile")
    def logfile():
        return jsonify(events=log.read(limit=100))

    return app
