from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from amir.common import (
    AI_LABOR_KEYWORDS,
    CONFIDENCE_TERMS,
    GUIDANCE_LOWERED_PATTERNS,
    GUIDANCE_RAISED_PATTERNS,
    build_transcript_metadata_csv,
    HEDGING_TERMS,
    METADATA_CSV_PATH,
    RISK_PATTERNS,
    SCORE_DB_PATH,
    SCORED_CSV_PATH,
    THEME_KEYWORDS,
    configure_logging,
    connect_sqlite,
    ensure_directories,
    init_score_db,
    load_app_config,
    parse_args,
    utc_now_iso,
)
from amir.nlp_utils import FinBERTScorer, LoughranMcDonaldLexicon, keyword_density, split_sentences

LOGGER = logging.getLogger(__name__)


def sentences_matching_keywords(text: str, keywords: list[str]) -> list[str]:
    matches = []
    patterns = [re.compile(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)") for keyword in keywords]
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(pattern.search(lowered) for pattern in patterns):
            matches.append(sentence)
    return matches


def aggregate_theme_score(finbert: FinBERTScorer, text: str, keywords: list[str]) -> float:
    matching = sentences_matching_keywords(text, keywords)
    if not matching:
        return 0.0
    summary = finbert.summarize(" ".join(matching))
    density = keyword_density(text, keywords)
    return density * summary["sentiment_score"]


def aggregate_ai_labor_score(finbert: FinBERTScorer, text: str) -> dict[str, float]:
    dimension_scores: dict[str, float] = {}
    for dimension, keywords in AI_LABOR_KEYWORDS.items():
        matching = sentences_matching_keywords(text, keywords)
        if not matching:
            dimension_scores[dimension] = 0.0
            continue
        dimension_scores[dimension] = finbert.summarize(" ".join(matching))["sentiment_score"]

    ai_labor_score = (
        dimension_scores["ai_tech"]
        + dimension_scores["labor_down"]
        - dimension_scores["labor_up"]
        + dimension_scores["productivity"]
    ) / 4

    return {**dimension_scores, "ai_labor_score": ai_labor_score}


def regex_flag(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def management_confidence_score(text: str) -> float:
    lowered = text.lower()
    confidence_hits = sum(
        len(re.findall(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered))
        for term in CONFIDENCE_TERMS
    )
    hedging_hits = sum(
        len(re.findall(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered))
        for term in HEDGING_TERMS
    )
    total = confidence_hits + hedging_hits
    if total == 0:
        return 0.0
    return (confidence_hits - hedging_hits) / total


def risk_mentions_count(text: str) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in RISK_PATTERNS)


def load_cached_scores(conn: sqlite3.Connection) -> dict[str, dict]:
    rows = conn.execute("SELECT transcript_id, payload_json FROM transcript_scores").fetchall()
    return {row["transcript_id"]: json.loads(row["payload_json"]) for row in rows}


def save_score(conn: sqlite3.Connection, transcript_id: str, payload: dict) -> None:
    conn.execute(
        """
        INSERT INTO transcript_scores (transcript_id, payload_json, scored_at)
        VALUES (?, ?, ?)
        ON CONFLICT(transcript_id) DO UPDATE SET payload_json=excluded.payload_json, scored_at=excluded.scored_at
        """,
        (transcript_id, json.dumps(payload), utc_now_iso()),
    )
    conn.commit()


def filter_metadata(metadata: pd.DataFrame, pattern: str | None = None, limit: int | None = None) -> pd.DataFrame:
    filtered = metadata
    if pattern:
        lowered = pattern.lower()
        relative_series = filtered.get("relative_path")
        if relative_series is not None:
            mask = relative_series.fillna("").astype(str).str.lower().str.contains(lowered, regex=False)
        else:
            mask = filtered["transcript_path"].fillna("").astype(str).str.lower().str.contains(lowered, regex=False)
        filtered = filtered.loc[mask]
    if limit is not None:
        filtered = filtered.head(limit)
    return filtered.reset_index(drop=True)


def resolve_transcript_path(row: pd.Series, transcripts_dir: Path) -> Path:
    raw_path = Path(str(row["transcript_path"]))
    if raw_path.exists():
        return raw_path

    parts = list(raw_path.parts)
    if "transcripts" in parts:
        index = parts.index("transcripts")
        candidate = transcripts_dir.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate

    relative_path = row.get("relative_path")
    if isinstance(relative_path, str) and relative_path:
        candidate = transcripts_dir / relative_path
        if candidate.exists():
            return candidate

    basename = raw_path.name
    matches = list(transcripts_dir.rglob(basename))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve transcript path for {row.get('transcript_id', basename)} "
        f"from metadata path {raw_path}"
    )


def score_transcript(finbert: FinBERTScorer, lm_lexicon: LoughranMcDonaldLexicon, row: pd.Series) -> dict:
    transcript_path = resolve_transcript_path(row, Path(row["transcripts_root"]))
    text = transcript_path.read_text(encoding="utf-8")
    sentiment = finbert.summarize(text)

    theme_scores = {
        f"theme_{theme}": aggregate_theme_score(finbert, text, keywords)
        for theme, keywords in THEME_KEYWORDS.items()
    }
    earnings_composite = sum(theme_scores.values()) / len(theme_scores)
    ai_scores = aggregate_ai_labor_score(finbert, text)
    lm_scores = lm_lexicon.score(text)

    payload = {
        **row.to_dict(),
        "transcript_path": str(transcript_path.resolve()),
        **sentiment,
        **theme_scores,
        "earnings_composite": earnings_composite,
        **ai_scores,
        **lm_scores,
        "guidance_raised": regex_flag(text, GUIDANCE_RAISED_PATTERNS),
        "guidance_lowered": regex_flag(text, GUIDANCE_LOWERED_PATTERNS),
        "management_confidence_score": management_confidence_score(text),
        "risk_mentions_count": risk_mentions_count(text),
        "scored_at": utc_now_iso(),
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = parse_args("Score transcripts with FinBERT and finance lexicons.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument("--force", action="store_true", help="Recompute scores for transcripts already cached.")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    ensure_directories(args.metadata_csv, args.output_csv, args.score_db)
    config = load_app_config(args.config)

    metadata_csv = args.metadata_csv
    if args.rebuild_metadata or not metadata_csv.exists():
        metadata_csv = build_transcript_metadata_csv(
            transcripts_dir=args.transcripts_dir,
            metadata_csv=metadata_csv,
            pattern=args.pattern,
            limit=args.limit,
        )

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    metadata = pd.read_csv(metadata_csv)
    metadata = filter_metadata(metadata, pattern=args.pattern, limit=args.limit)
    conn = connect_sqlite(args.score_db)
    init_score_db(conn)
    cached = load_cached_scores(conn)

    finbert = FinBERTScorer(
        model_name=config.finbert_model_name,
        batch_size=config.batch_size,
        max_sentences=config.max_sentences,
    )
    lm_lexicon = LoughranMcDonaldLexicon()

    output_rows: list[dict] = []
    iterator = tqdm(metadata.to_dict(orient="records"), desc="Scoring transcripts")
    for raw_row in iterator:
        row = pd.Series({**raw_row, "transcripts_root": str(args.transcripts_dir.resolve())})
        transcript_id = row["transcript_id"]
        if transcript_id in cached and not args.force:
            output_rows.append(cached[transcript_id])
            continue

        payload = score_transcript(finbert, lm_lexicon, row)
        save_score(conn, transcript_id, payload)
        output_rows.append(payload)

    if output_rows:
        scored = pd.DataFrame(output_rows).sort_values(["call_date", "ticker"], ascending=[False, True])
    else:
        scored = pd.DataFrame(columns=metadata.columns.tolist())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(scored):,} scored transcripts to {args.output_csv}")


if __name__ == "__main__":
    main()
