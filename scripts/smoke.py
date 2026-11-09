"""Exercise the HTTP API from outside the running server."""

import json
import os
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

url = os.getenv("FORECAST_URL", "http://127.0.0.1:8000")


def get(path, **params):
    with urlopen(f"{url}{path}?{urlencode(params)}", timeout=3) as response:
        return json.load(response)


for _ in range(30):
    try:
        health = get("/health")
        break
    except (URLError, ConnectionError, TimeoutError):
        time.sleep(1)
else:
    raise SystemExit("Forecast API did not become ready")

assert health["status"] == "ready"
for country in ("United Kingdom", "all"):
    result = get("/predict", country=country, as_of="2019-07-31")
    assert result["country"] == country.lower(), result
    assert result["forecast_end"] == "2019-08-30", result
    assert isinstance(result["predicted_revenue"], (int, float)), result
    print(result)
assert len(get("/logfile")["events"]) >= 2
print("HTTP smoke checks passed")
