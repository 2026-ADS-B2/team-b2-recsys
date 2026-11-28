import time
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CATALOG_PATH = Path("data/processed/catalog.parquet")
SECRETS_PATH = Path(".streamlit/secrets.toml")
PLACEHOLDER_IMAGE = "https://placehold.co/300x450?text=No+poster"
TMDB_API_URL = "https://api.themoviedb.org/3"


def main() -> None:
    with SECRETS_PATH.open("rb") as secrets_file:
        token = tomllib.load(secrets_file)["TMDB_READ_ACCESS_TOKEN"]

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        }
    )

    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    config_response = session.get(f"{TMDB_API_URL}/configuration", timeout=20)
    config_response.raise_for_status()
    image_config = config_response.json()["images"]
    image_base_url = image_config["secure_base_url"]
    poster_size = "w500" if "w500" in image_config["poster_sizes"] else "original"

    catalog = pd.read_parquet(CATALOG_PATH)
    has_tmdb_url = catalog["tmdb_url"].map(
        lambda value: isinstance(value, str) and bool(value)
    )
    needs_poster = catalog["image_url"].eq(PLACEHOLDER_IMAGE)
    rows_to_check = catalog[has_tmdb_url & needs_poster]
    total = len(rows_to_check)

    for number, (row_index, movie) in enumerate(rows_to_check.iterrows(), start=1):
        tmdb_id = urlparse(movie["tmdb_url"]).path.rstrip("/").rsplit("/", 1)[-1]
        response = session.get(
            f"{TMDB_API_URL}/movie/{tmdb_id}",
            params={"language": "en-US"},
            timeout=20,
        )

        if response.status_code == 404:
            time.sleep(0.1)
            continue

        response.raise_for_status()
        poster_path = response.json().get("poster_path")

        if poster_path:
            catalog.at[row_index, "image_url"] = (
                f"{image_base_url}{poster_size}{poster_path}"
            )

        time.sleep(0.1)

        if number % 100 == 0:
            catalog.to_parquet(CATALOG_PATH, index=False)
            print(f"Checked {number} of {total} movies")

    catalog.to_parquet(CATALOG_PATH, index=False)
    poster_count = int(catalog["image_url"].ne(PLACEHOLDER_IMAGE).sum())
    print(f"Done. Catalog now has {poster_count} poster URLs.")


if __name__ == "__main__":
    main()