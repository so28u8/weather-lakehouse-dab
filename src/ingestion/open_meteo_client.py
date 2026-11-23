import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CITIES = [
    {"name": "Budapest", "country": "HU", "lat": 47.4979, "lon": 19.0402},
    {"name": "Vienna", "country": "AT", "lat": 48.2082, "lon": 16.3738},
    {"name": "Frankfurt", "country": "DE", "lat": 50.1109, "lon": 8.6821},
    {"name": "Berlin", "country": "DE", "lat": 52.5200, "lon": 13.4050},
    {"name": "Prague", "country": "CZ", "lat": 50.0755, "lon": 14.4378},
]

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        respect_retry_after_header=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def fetch_city_weather(session: requests.Session, city: dict[str, Any]) -> dict[str, Any]:
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
        "past_days": 10,
        "forecast_days": 1,
    }
    print(f"Lekérés indítása: {city['name']}...", flush=True)
    response = session.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    print(f"Kész: {city['name']} ({response.status_code})", flush=True)

    payload = response.json()
    payload["_metadata"] = {
        "city": city["name"],
        "country": city["country"],
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "open-meteo-forecast-v1",
    }
    return payload

def run_ingestion(landing_path: str) -> None:
    session = build_session()
    now = datetime.now(timezone.utc)
    partition_dir = Path(landing_path) / f"ingest_date={now.strftime('%Y-%m-%d')}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    for city in CITIES:
        logger.info("Fetching weather data for %s...", city["name"])
        data = fetch_city_weather(session, city)
        file_path = partition_dir / f"{city['name'].lower()}_{timestamp}.json"
        file_path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Saved %s", file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Open-Meteo weather data into UC Volume.")
    parser.add_argument("--landing-path", required=True, help="Target Unity Catalog Volume path")
    args = parser.parse_args()
    run_ingestion(args.landing_path)


if __name__ == "__main__":
    main()