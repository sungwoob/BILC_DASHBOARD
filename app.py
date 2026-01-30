from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import ast
from collections import Counter
from typing import List, Any

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
    has_function_list: bool = False
    function_name_unique_count: int | None = None
    function_name_values: List[str] | None = None
    function_name_counts: List[tuple[str, int]] | None = None
    normal_qa_count: int | None = None
    error: str | None = None


def summarize_csv(path: Path) -> CsvSummary:
    try:
        dataframe = pd.read_csv(path)
        has_function_list = "functionList" in dataframe.columns
        function_name_unique_count = None
        function_name_values: List[str] | None = None
        function_name_counts: List[tuple[str, int]] | None = None
        if has_function_list:
            names: List[str] = []
            normal_qa_count = 0

            def extract_function_names(parsed: Any) -> List[str]:
                extracted: List[str] = []
                if isinstance(parsed, dict):
                    function_value = parsed.get("function")
                    if isinstance(function_value, dict):
                        function_name = function_value.get("name")
                        if isinstance(function_name, str):
                            extracted.append(function_name)
                    functions_value = parsed.get("functions")
                    if isinstance(functions_value, list):
                        for item in functions_value:
                            if not isinstance(item, dict):
                                continue
                            function_name = item.get("name")
                            if isinstance(function_name, str):
                                extracted.append(function_name)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        function_value = item.get("function", item)
                        if isinstance(function_value, dict):
                            function_name = function_value.get("name")
                            if isinstance(function_name, str):
                                extracted.append(function_name)
                return extracted

            for value in dataframe["functionList"].dropna().astype(str):
                parsed = None
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        parsed = None
                extracted_names = extract_function_names(parsed)
                if extracted_names:
                    names.extend(extracted_names)
                else:
                    normal_qa_count += 1
            name_counts = Counter(names)
            unique_names = sorted(name_counts.keys())
            function_name_unique_count = len(unique_names)
            function_name_values = unique_names
            function_name_counts = [
                (name, name_counts[name]) for name in unique_names
            ]
        return CsvSummary(
            filename=path.name,
            row_count=int(dataframe.shape[0]),
            column_count=int(dataframe.shape[1]),
            columns=[str(column) for column in dataframe.columns],
            has_function_list=has_function_list,
            function_name_unique_count=function_name_unique_count,
            function_name_values=function_name_values,
            function_name_counts=function_name_counts,
            normal_qa_count=normal_qa_count if has_function_list else None,
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
    app.run(host="0.0.0.0", port=8000, debug=True)
