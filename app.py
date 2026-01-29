from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from flask import Flask, abort, render_template, send_from_directory

APP_ROOT = Path(__file__).resolve().parent
DATASET_DIR = APP_ROOT / "dataset"


@dataclass
class CsvSummary:
    filename: str
    row_count: int
    column_count: int
    columns: List[str]
    error: str | None = None


def summarize_csv(path: Path) -> CsvSummary:
    try:
        dataframe = pd.read_csv(path)
        return CsvSummary(
            filename=path.name,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            columns=[str(column) for column in dataframe.columns],
        )
    except Exception as exc:  # noqa: BLE001 - show error in UI
        return CsvSummary(
            filename=path.name,
            row_count=0,
            column_count=0,
            columns=[],
            error=str(exc),
        )


def load_summaries() -> List[CsvSummary]:
    if not DATASET_DIR.exists():
        return []

    csv_files = sorted(DATASET_DIR.glob("*.csv"))
    return [summarize_csv(path) for path in csv_files]


app = Flask(__name__)


@app.route("/")
def index() -> str:
    summaries = load_summaries()
    return render_template(
        "index.html",
        summaries=summaries,
        dataset_dir=str(DATASET_DIR),
    )


@app.route("/download/<path:filename>")
def download_csv(filename: str):
    file_path = DATASET_DIR / filename
    if not file_path.exists() or file_path.suffix.lower() != ".csv":
        return abort(404)
    return send_from_directory(DATASET_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
