"""
Data Loader Module.

Handles automated downloading, decompression, and raw data loading
from the UCI Machine Learning Repository.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

from src.config import (
    DATASET_URL,
    FALLBACK_URL,
    RAW_DATA_DIR,
    RAW_HOUR_FILE,
    RAW_DAY_FILE,
    RAW_ZIP_FILE,
    REPORTS_DIR,
)
from src.utils import setup_logger, save_json

logger = setup_logger("data_loader")

DATASET_METADATA = {
    "name": "UCI Bike Sharing Dataset",
    "source_url": DATASET_URL,
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0) / Citation Required",
    "citation": (
        "Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors "
        "and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, "
        "Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3."
    ),
    "target_variable": "cnt",
    "target_description": "Total hourly count of rental bikes (sum of casual and registered users)",
    "granularity": "Hourly (17,379 observations) and Daily (731 observations)",
    "timeframe": "2011-01-01 00:00:00 to 2012-12-31 23:00:00 (2 years)",
    "feature_descriptions": {
        "instant": "Record index (sequential)",
        "dteday": "Date of observation (YYYY-MM-DD)",
        "season": "Season (1: Winter/Spring start, 2: Summer, 3: Fall, 4: Winter)",
        "yr": "Year (0: 2011, 1: 2012)",
        "mnth": "Month of the year (1 to 12)",
        "hr": "Hour of the day (0 to 23)",
        "holiday": "Whether the day is a holiday (1: Yes, 0: No)",
        "weekday": "Day of the week (0: Sunday to 6: Saturday)",
        "workingday": "Working day indicator (1: Neither weekend nor holiday, 0: Weekend or holiday)",
        "weathersit": "Weather category (1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Ice/Fog)",
        "temp": "Normalized temperature in Celsius (divided by 41)",
        "atemp": "Normalized feeling temperature in Celsius (divided by 50)",
        "hum": "Normalized relative humidity (divided by 100)",
        "windspeed": "Normalized wind speed (divided by 67)",
        "casual": "Count of unregistered casual users (leakage component)",
        "registered": "Count of registered annual/monthly members (leakage component)",
        "cnt": "Count of total rental bikes (prediction target)"
    },
    "limitations": [
        "Aggregated city-level data for Washington, D.C.; cannot resolve spatial micro-movements.",
        "Occasional missing hours during extreme weather events or late night maintenance.",
        "Weather attributes are pre-normalized in the original UCI release.",
        "Strong structural break / secular growth between 2011 and 2012 due to system adoption."
    ]
}


class DataLoader:
    """Automated dataset downloader and loader with fallback mechanisms."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.raw_dir / "bike_sharing_dataset.zip"
        self.hour_path = self.raw_dir / "hour.csv"
        self.day_path = self.raw_dir / "day.csv"

    def download_dataset(self, force: bool = False) -> Path:
        """
        Download the UCI Bike Sharing Dataset archive if not already cached.
        """
        if self.hour_path.exists() and not force:
            logger.info("Raw dataset already exists at %s. Skipping download.", self.hour_path)
            return self.hour_path

        urls = [DATASET_URL, FALLBACK_URL]
        downloaded = False

        for url in urls:
            try:
                logger.info("Attempting to download dataset from %s ...", url)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnsembleForecaster/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read()
                    with open(self.zip_path, "wb") as f:
                        f.write(content)

                logger.info("Decompressing zip archive into %s ...", self.raw_dir)
                with zipfile.ZipFile(self.zip_path, "r") as archive:
                    archive.extractall(self.raw_dir)

                downloaded = True
                logger.info("Successfully downloaded and extracted dataset.")
                break
            except Exception as e:
                logger.warning("Failed downloading from %s: %s", url, e)

        if not downloaded:
            raise RuntimeError(
                f"Could not download dataset from any designated source. "
                f"Please ensure network access or manually place hour.csv into {self.raw_dir}"
            )

        return self.hour_path

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the original raw hourly dataset into a pandas DataFrame.
        """
        if not self.hour_path.exists():
            self.download_dataset()

        logger.info("Loading raw hourly data from %s", self.hour_path)
        df = pd.read_csv(self.hour_path)
        logger.info("Loaded raw dataset with shape: %s", df.shape)
        return df

    def export_documentation(self, output_dir: Path = REPORTS_DIR) -> Path:
        """Export dataset documentation report in markdown and json formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "dataset_metadata.json"
        save_json(DATASET_METADATA, json_path)

        md_path = output_dir / "dataset_documentation.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Dataset Documentation: {DATASET_METADATA['name']}\n\n")
            f.write(f"- **Source URL**: {DATASET_METADATA['source_url']}\n")
            f.write(f"- **License**: {DATASET_METADATA['license']}\n")
            f.write(f"- **Citation**: {DATASET_METADATA['citation']}\n")
            f.write(f"- **Target Variable**: `{DATASET_METADATA['target_variable']}` ({DATASET_METADATA['target_description']})\n")
            f.write(f"- **Granularity**: {DATASET_METADATA['granularity']}\n")
            f.write(f"- **Timeframe**: {DATASET_METADATA['timeframe']}\n\n")

            f.write("## Feature Schema\n\n")
            f.write("| Column Name | Type | Description |\n")
            f.write("|-------------|------|-------------|\n")
            for col, desc in DATASET_METADATA["feature_descriptions"].items():
                f.write(f"| `{col}` | Feature | {desc} |\n")

            f.write("\n## Known Dataset Limitations\n\n")
            for lim in DATASET_METADATA["limitations"]:
                f.write(f"- {lim}\n")

        logger.info("Saved dataset documentation to %s and %s", json_path, md_path)
        return md_path

