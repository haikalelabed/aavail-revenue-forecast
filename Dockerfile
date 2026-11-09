FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/runtime/model.joblib LOG_PATH=/app/runtime/predictions.sqlite \
    TRAINING_DAILY_PATH=/app/data/train.csv OBSERVED_DAILY_PATH=/app/data/observed.csv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --system --uid 10001 --create-home forecast \
    && mkdir -p runtime && chown forecast runtime
COPY forecast/ forecast/
COPY tests/ tests/
COPY scripts/run_tests.sh scripts/run_tests.sh
COPY data/train.csv data/observed.csv data/
USER forecast
RUN python -m forecast.model "$TRAINING_DAILY_PATH" "$MODEL_PATH"
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "forecast.api:create_app()"]
