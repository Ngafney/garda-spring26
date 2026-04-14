from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
METADATA_CSV_PATH = DEFAULT_OUTPUT_DIR / "transcript_metadata.csv"
SCORE_DB_PATH = DEFAULT_OUTPUT_DIR / "score_cache.sqlite"
SCORED_CSV_PATH = DEFAULT_OUTPUT_DIR / "transcript_scores.csv"

THEME_KEYWORDS = {
    "demand": ["demand", "orders", "consumption", "spending", "volume", "backlog"],
    "pricing": ["price", "pricing", "inflation", "discount", "margin pressure", "cost"],
    "guidance": ["guidance", "outlook", "forecast", "expect", "raise", "lower"],
    "labor": ["labor", "hiring", "headcount", "staffing", "wage", "employment"],
    "capex": ["capex", "capital expenditure", "investment", "facility", "infrastructure"],
    "ai": ["ai", "artificial intelligence", "automation", "machine learning", "copilot"],
}

AI_LABOR_KEYWORDS = {
    "ai_tech": ["ai", "artificial intelligence", "automation", "machine learning", "copilot"],
    "labor_down": ["headcount reduction", "layoff", "downsizing", "labor savings", "fewer workers"],
    "labor_up": ["hiring", "added employees", "headcount growth", "staffing up", "recruitment"],
    "productivity": ["productivity", "efficiency", "streamline", "automation gains", "throughput"],
}

CONFIDENCE_TERMS = [
    "confident",
    "confidence",
    "strong",
    "optimistic",
    "encouraged",
    "well positioned",
    "solid",
    "momentum",
]

HEDGING_TERMS = [
    "uncertain",
    "uncertainty",
    "cautious",
    "challenging",
    "volatility",
    "risk",
    "headwind",
    "pressure",
]

GUIDANCE_RAISED_PATTERNS = [
    r"\brais(?:e|ed|ing)\b.{0,40}\bguidance\b",
    r"\bincreas(?:e|ed|ing)\b.{0,40}\boutlook\b",
    r"\bupdat(?:e|ed|ing)\b.{0,40}\bupward\b",
]

GUIDANCE_LOWERED_PATTERNS = [
    r"\blower(?:ed|ing)?\b.{0,40}\bguidance\b",
    r"\breduc(?:e|ed|ing)\b.{0,40}\boutlook\b",
    r"\bcut\b.{0,40}\bguidance\b",
]

RISK_PATTERNS = [
    r"\brisk\b",
    r"\brisks\b",
    r"\bheadwind\b",
    r"\bheadwinds\b",
    r"\bpressure\b",
    r"\bvolatility\b",
    r"\buncertain(?:ty)?\b",
]

TRANSCRIPT_STEM_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9\.-]+)_(?P<call_date>\d{4}-\d{2}-\d{2})_(?P<fiscal_period>[^.]+)$"
)


@dataclass(slots=True)
class AppConfig:
    finbert_model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    max_sentences: int | None = 160


def configure_logging(verbose: int) -> None:
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def parse_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Root directory containing transcript .txt files.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV_PATH,
        help="Metadata CSV used to drive scoring.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCORED_CSV_PATH,
        help="Where to write the scored transcript CSV.",
    )
    parser.add_argument(
        "--score-db",
        type=Path,
        default=SCORE_DB_PATH,
        help="SQLite cache used for transcript scoring results.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter applied to transcript paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of transcript files included.",
    )
    parser.add_argument(
        "--rebuild-metadata",
        action="store_true",
        help="Rebuild the metadata CSV before scoring.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity.")
    return parser


def load_app_config(path: Path | None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return AppConfig(
        finbert_model_name=raw.get("finbert_model_name", config.finbert_model_name),
        batch_size=int(raw.get("batch_size", config.batch_size)),
        max_sentences=raw.get("max_sentences", config.max_sentences),
    )


def ensure_directories(*paths: Path) -> None:
    targets = paths or (DEFAULT_OUTPUT_DIR,)
    for path in targets:
        directory = path if path.suffix == "" else path.parent
        directory.mkdir(parents=True, exist_ok=True)


def connect_sqlite(path: Path) -> sqlite3.Connection:
    ensure_directories(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_score_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_scores (
            transcript_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            scored_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug_to_company_name(slug: str) -> str:
    cleaned = slug.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", cleaned).strip().title()


def transcript_metadata_for_path(path: Path, transcripts_dir: Path) -> dict[str, str]:
    relative_path = path.relative_to(transcripts_dir)
    match = TRANSCRIPT_STEM_RE.match(path.stem)

    company_slug = relative_path.parts[0] if len(relative_path.parts) >= 1 else path.parent.name
    fiscal_year = relative_path.parts[1] if len(relative_path.parts) >= 2 else ""
    fiscal_period = relative_path.parts[2] if len(relative_path.parts) >= 3 else ""
    ticker = match.group("ticker") if match else path.stem.split("_")[0]
    call_date = match.group("call_date") if match else ""
    fiscal_period = match.group("fiscal_period") if match else fiscal_period or ""

    return {
        "transcript_id": relative_path.as_posix().removesuffix(path.suffix),
        "company_slug": company_slug,
        "company_name": slug_to_company_name(company_slug),
        "ticker": ticker,
        "call_date": call_date,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "transcript_path": str(path.resolve()),
        "relative_path": relative_path.as_posix(),
    }


def build_transcript_metadata_csv(
    transcripts_dir: Path,
    metadata_csv: Path,
    pattern: str | None = None,
    limit: int | None = None,
) -> Path:
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    ensure_directories(metadata_csv)
    needle = pattern.lower() if pattern else None
    rows: list[dict[str, str]] = []

    for path in sorted(transcripts_dir.rglob("*.txt")):
        relative_text = path.relative_to(transcripts_dir).as_posix().lower()
        if needle and needle not in relative_text:
            continue
        rows.append(transcript_metadata_for_path(path, transcripts_dir))
        if limit is not None and len(rows) >= limit:
            break

    fieldnames = [
        "transcript_id",
        "company_slug",
        "company_name",
        "ticker",
        "call_date",
        "fiscal_year",
        "fiscal_period",
        "transcript_path",
        "relative_path",
    ]
    with metadata_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Wrote %s transcript metadata rows to %s", len(rows), metadata_csv)
    return metadata_csv
