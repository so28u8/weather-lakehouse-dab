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

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


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


def fetch_city_air_quality(session: requests.Session, city: dict[str, Any]) -> dict[str, Any]:
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": "pm10,pm2_5,nitrogen_dioxide,european_aqi",
        "timezone": "UTC",
        "past_days": 5,
        "forecast_days": 2,
    }
    print(f"Air Quality lekérés indítása: {city['name']}...", flush=True)
    response = session.get(AIR_QUALITY_URL, params=params, timeout=(5, 15))
    response.raise_for_status()
    print(f"Kész: {city['name']} ({response.status_code})", flush=True)

    payload = response.json()
    payload["_metadata"] = {
        "city": city["name"],
        "country": city["country"],
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "open-meteo-air-quality-v1",
    }
    return payload


def save_payload(payload: dict[str, Any], landing_path: Path, city_name: str) -> Path:
    landing_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = landing_path / f"air_quality_{city_name.lower()}_{ts}.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Open-Meteo Air Quality data.")
    parser.add_argument("--landing-path", required=True, help="Target Unity Catalog Volume path")
    args = parser.parse_args()

    landing_path = Path(args.landing_path)
    session = build_session()

    for city in CITIES:
        payload = fetch_city_air_quality(session, city)
        out_file = save_payload(payload, landing_path, city["name"])
        logger.info("Saved %s -> %s", city["name"], out_file)


if __name__ == "__main__":
    main()