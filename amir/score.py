from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import (
    AI_LABOR_KEYWORDS,
    CONFIDENCE_TERMS,
    GUIDANCE_LOWERED_PATTERNS,
    GUIDANCE_RAISED_PATTERNS,
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
from nlp_utils import FinBERTScorer, LoughranMcDonaldLexicon, keyword_density, split_sentences

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


def score_transcript(finbert: FinBERTScorer, lm_lexicon: LoughranMcDonaldLexicon, row: pd.Series) -> dict:
    text = Path(row["transcript_path"]).read_text(encoding="utf-8")
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
    ensure_directories()
    config = load_app_config(args.config)

    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV_PATH}")

    metadata = pd.read_csv(METADATA_CSV_PATH)
    conn = connect_sqlite(SCORE_DB_PATH)
    init_score_db(conn)
    cached = load_cached_scores(conn)

    finbert = FinBERTScorer(model_name=config.finbert_model_name)
    lm_lexicon = LoughranMcDonaldLexicon()

    output_rows: list[dict] = []
    iterator = tqdm(metadata.to_dict(orient="records"), desc="Scoring transcripts")
    for raw_row in iterator:
        row = pd.Series(raw_row)
        transcript_id = row["transcript_id"]
        if transcript_id in cached and not args.force:
            output_rows.append(cached[transcript_id])
            continue

        payload = score_transcript(finbert, lm_lexicon, row)
        save_score(conn, transcript_id, payload)
        output_rows.append(payload)

    scored = pd.DataFrame(output_rows).sort_values(["call_date", "ticker"], ascending=[False, True])
    SCORED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_CSV_PATH, index=False)
    print(f"Wrote {len(scored):,} scored transcripts to {SCORED_CSV_PATH}")


if __name__ == "__main__":
    main()
