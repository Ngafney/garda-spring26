from __future__ import annotations

"""
Lightweight earnings-call sentiment panel.

Design choices are intentionally simple and scalable:
- finance-flavored lexicon tone instead of transformer inference
- prepared-remarks versus Q&A separation
- analyst-versus-management Q&A tone gap where the transcript format allows it
- topic signals for demand, pricing, capex, labor, automation, and macro risk
- company and aggregate time-series rollups for macro and trading research
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "research_sentiment"

WORD_RE = re.compile(r"\b[\w%./-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DATE_IN_STEM_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<call_date>\d{4}-\d{2}-\d{2})_(?P<fiscal_period>[^.]+)$")
QA_HEADER_RE = re.compile(
    r"^\s*(questions?\s*(?:&|and)\s*answers?|question-?and-?answer(?:\s+session)?|q&a)\s*:?\s*$",
    flags=re.IGNORECASE,
)
QA_TRANSITION_RE = re.compile(
    r"\b("
    r"we will now begin the question-and-answer session|"
    r"we will now be conducting a question-and-answer session|"
    r"open the line for q&a|"
    r"open the call for q&a|"
    r"operator[, ]+please open the call for q&a|"
    r"operator[, ]+please provide instructions for those interested in asking a question|"
    r"let'?s open it up for questions"
    r")\b",
    flags=re.IGNORECASE,
)
MANAGEMENT_TITLE_HINTS = (
    "chief",
    "ceo",
    "cfo",
    "coo",
    "president",
    "chair",
    "chairman",
    "founder",
    "investor relations",
    "finance",
    "financial officer",
    "treasurer",
    "controller",
    "vice president",
    "svp",
    "evp",
    "vp",
    "director",
    "head of",
    "general manager",
)
ANALYST_TITLE_HINTS = (
    "analyst",
    "research",
    "securities",
    "capital markets",
    "equity research",
)
BOILERPLATE_PATTERNS = (
    re.compile(r"\bforward-looking statements?\b", flags=re.IGNORECASE),
    re.compile(r"\bsafe harbor\b", flags=re.IGNORECASE),
    re.compile(r"\bnon-gaap\b", flags=re.IGNORECASE),
    re.compile(r"\breconciliation\b", flags=re.IGNORECASE),
    re.compile(r"\boperator instructions?\b", flags=re.IGNORECASE),
    re.compile(r"\binvestor relations website\b", flags=re.IGNORECASE),
    re.compile(r"\bsec\b", flags=re.IGNORECASE),
    re.compile(r"\bwebcast replay\b", flags=re.IGNORECASE),
    re.compile(r"\bpress release\b", flags=re.IGNORECASE),
)
GUIDANCE_RAISED_PATTERNS = (
    re.compile(r"\brais(?:e|ed|ing)\b.{0,40}\bguidance\b", flags=re.IGNORECASE),
    re.compile(r"\bincreas(?:e|ed|ing)\b.{0,40}\boutlook\b", flags=re.IGNORECASE),
    re.compile(r"\bupdat(?:e|ed|ing)\b.{0,40}\bupward\b", flags=re.IGNORECASE),
)
GUIDANCE_LOWERED_PATTERNS = (
    re.compile(r"\blower(?:ed|ing)?\b.{0,40}\bguidance\b", flags=re.IGNORECASE),
    re.compile(r"\breduc(?:e|ed|ing)\b.{0,40}\boutlook\b", flags=re.IGNORECASE),
    re.compile(r"\bcut\b.{0,40}\bguidance\b", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class Block:
    section: str
    speaker: str | None
    title: str | None
    text: str


@dataclass(frozen=True)
class LexiconSpec:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


class PhraseCounter:
    def __init__(self, phrases: Sequence[str]):
        escaped = [re.escape(phrase.lower()) for phrase in phrases if phrase]
        self.pattern = (
            re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), flags=re.IGNORECASE)
            if escaped
            else None
        )

    def count(self, text: str) -> int:
        if not text or self.pattern is None:
            return 0
        return len(self.pattern.findall(text))


BASE_SENTIMENT = LexiconSpec(
    positive=(
        "accelerating",
        "accretive",
        "backlog",
        "benefit",
        "beneficial",
        "beat",
        "confidence",
        "confident",
        "constructive",
        "disciplined",
        "durable",
        "efficiency",
        "efficient",
        "expand",
        "expansion",
        "favorable",
        "free cash flow",
        "gain share",
        "growth",
        "healthy",
        "improve",
        "improved",
        "improving",
        "margin expansion",
        "momentum",
        "opportunity",
        "outperform",
        "positive",
        "pricing power",
        "productivity",
        "ramp",
        "record",
        "resilient",
        "robust",
        "solid",
        "stable",
        "strength",
        "strong",
        "upside",
        "well positioned",
    ),
    negative=(
        "challenging",
        "compression",
        "constraint",
        "constraints",
        "cautious",
        "cut",
        "cutting",
        "decline",
        "deceleration",
        "destocking",
        "deterioration",
        "difficult",
        "disruption",
        "downturn",
        "erosion",
        "headwind",
        "impairment",
        "inflationary",
        "loss",
        "miss",
        "negative",
        "pressure",
        "recession",
        "restructuring",
        "risk",
        "shortage",
        "slowdown",
        "soft",
        "softness",
        "uncertain",
        "uncertainty",
        "underperform",
        "volatility",
        "weak",
        "weaker",
        "weakness",
    ),
)
UNCERTAINTY_TERMS = (
    "uncertain",
    "uncertainty",
    "volatile",
    "volatility",
    "visibility",
    "limited visibility",
    "challenging backdrop",
    "challenging environment",
    "macro uncertainty",
    "not clear",
    "unknown",
    "range of outcomes",
    "hard to predict",
    "difficult to predict",
    "fluid environment",
    "monitor closely",
    "cautious",
)
RISK_TERMS = (
    "headwind",
    "headwinds",
    "pressure",
    "risk",
    "risks",
    "challenging",
    "volatility",
    "recession",
    "tariff",
    "tariffs",
    "geopolitical",
    "fx",
    "foreign exchange",
    "interest rate",
    "interest rates",
    "consumer weakness",
    "slowdown",
    "macro pressure",
    "uncertainty",
)
DEMAND_LEXICON = LexiconSpec(
    positive=(
        "strong demand",
        "healthy demand",
        "robust demand",
        "solid demand",
        "better demand",
        "improving demand",
        "stable demand",
        "order growth",
        "bookings growth",
        "backlog growth",
        "good demand",
        "demand recovery",
        "volume growth",
        "share gains",
        "market share gains",
    ),
    negative=(
        "weak demand",
        "soft demand",
        "demand slowdown",
        "slowing demand",
        "lower demand",
        "demand pressure",
        "order weakness",
        "bookings weakness",
        "backlog pressure",
        "customer caution",
        "cautious customer",
        "destocking",
        "inventory correction",
        "volume pressure",
        "traffic weakness",
    ),
)
PRICING_LEXICON = LexiconSpec(
    positive=(
        "pricing power",
        "price increase",
        "price increases",
        "positive pricing",
        "favorable pricing",
        "pricing discipline",
        "price realization",
        "net price",
        "margin expansion",
        "mix benefit",
        "premiumization",
        "higher price",
        "pass-through",
        "pass through",
        "pricing actions",
    ),
    negative=(
        "price pressure",
        "pricing pressure",
        "promotional",
        "promotions",
        "discounting",
        "discounts",
        "margin pressure",
        "cost inflation",
        "inflationary pressure",
        "input cost",
        "commodity inflation",
        "mix headwind",
        "unfavorable mix",
        "deflation",
        "price elasticity",
    ),
)
CAPEX_LEXICON = LexiconSpec(
    positive=(
        "capital expenditure",
        "capex",
        "investment",
        "investing",
        "capacity expansion",
        "buildout",
        "factory expansion",
        "new plant",
        "greenfield",
        "brownfield",
        "data center",
        "expansion project",
        "ramping capacity",
        "automation investment",
        "infrastructure investment",
    ),
    negative=(
        "cut capex",
        "reduce capex",
        "lower capex",
        "pause investment",
        "delay investment",
        "project delay",
        "project delays",
        "cancel project",
        "capacity reduction",
    ),
)
SUPPLY_PRESSURE_TERMS = (
    "supply chain",
    "shortage",
    "shortages",
    "constraint",
    "constraints",
    "constrained",
    "bottleneck",
    "bottlenecks",
    "lead time",
    "lead times",
    "backorder",
    "backorders",
    "shipping delay",
    "freight pressure",
    "logistics pressure",
    "inventory shortage",
    "component shortage",
)
LABOR_PRESSURE_TERMS = (
    "labor shortage",
    "tight labor market",
    "labor availability",
    "wage pressure",
    "hiring challenge",
    "staffing challenge",
    "recruiting challenge",
    "turnover",
    "retention challenge",
    "overtime",
    "labor inflation",
    "wage inflation",
    "headcount pressure",
)
AUTOMATION_TERMS = (
    "automation",
    "automate",
    "automated",
    "robotics",
    "ai",
    "artificial intelligence",
    "machine learning",
    "copilot",
    "productivity gains",
    "efficiency gains",
    "digitalization",
    "software driven",
    "self-service",
)
MACRO_RISK_TERMS = (
    "recession",
    "macro uncertainty",
    "geopolitical",
    "tariff",
    "tariffs",
    "interest rate",
    "interest rates",
    "higher rates",
    "consumer weakness",
    "europe weakness",
    "china weakness",
    "fx headwind",
    "foreign exchange",
    "currency headwind",
    "inflation",
    "deflation",
    "credit tightening",
)

COUNTERS = {
    "base_positive": PhraseCounter(BASE_SENTIMENT.positive),
    "base_negative": PhraseCounter(BASE_SENTIMENT.negative),
    "uncertainty": PhraseCounter(UNCERTAINTY_TERMS),
    "risk": PhraseCounter(RISK_TERMS),
    "demand_positive": PhraseCounter(DEMAND_LEXICON.positive),
    "demand_negative": PhraseCounter(DEMAND_LEXICON.negative),
    "pricing_positive": PhraseCounter(PRICING_LEXICON.positive),
    "pricing_negative": PhraseCounter(PRICING_LEXICON.negative),
    "capex_positive": PhraseCounter(CAPEX_LEXICON.positive),
    "capex_negative": PhraseCounter(CAPEX_LEXICON.negative),
    "supply_pressure": PhraseCounter(SUPPLY_PRESSURE_TERMS),
    "labor_pressure": PhraseCounter(LABOR_PRESSURE_TERMS),
    "automation": PhraseCounter(AUTOMATION_TERMS),
    "macro_risk": PhraseCounter(MACRO_RISK_TERMS),
}

ROLLING_SIGNAL_COLUMNS = (
    "overall_net_tone",
    "prepared_net_tone",
    "management_qa_net_tone",
    "analyst_question_net_tone",
    "demand_net",
    "pricing_power_net",
    "capex_net",
    "supply_chain_pressure_density",
    "labor_pressure_density",
    "automation_density",
    "uncertainty_density",
    "macro_risk_density",
    "growth_signal",
    "margin_signal",
    "credibility_signal",
    "composite_signal",
)
AGGREGATION_COLUMNS = (
    "overall_net_tone",
    "prepared_net_tone",
    "qa_net_tone",
    "management_qa_net_tone",
    "analyst_question_net_tone",
    "demand_net",
    "pricing_power_net",
    "capex_net",
    "supply_chain_pressure_density",
    "labor_pressure_density",
    "automation_density",
    "uncertainty_density",
    "macro_risk_density",
    "growth_signal",
    "margin_signal",
    "credibility_signal",
    "composite_signal",
    "qna_reality_gap",
    "analyst_management_gap",
    "complexity_score",
    "numeric_token_share",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight, research-inspired earnings-call signal panel using "
            "lexicons and section-aware transcript parsing. This script does not use FinBERT."
        )
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Root folder containing transcript .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where panel and macro CSVs will be written.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter on relative transcript paths.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of transcript files to score.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use for transcript scoring.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output files in the target output directory before scoring.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Emit a terminal checkpoint after this many newly appended transcript rows.",
    )
    return parser.parse_args()


def configure_logging(verbose: int) -> None:
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slug_to_name(slug: str) -> str:
    cleaned = slug.replace("-", " ").replace("_", " ").strip()
    return re.sub(r"\s+", " ", cleaned).title()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences = [normalize_space(part) for part in SENTENCE_RE.split(text) if normalize_space(part)]
    return sentences


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def safe_density(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return count / denominator


def safe_net(positive: int, negative: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (positive - negative) / denominator


def rolling_zscore(series: pd.Series, window: int = 8, min_periods: int = 4) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    zscore = (series - rolling_mean) / rolling_std
    return zscore.fillna(0.0)


def looks_like_speaker_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90 or stripped.endswith(":"):
        return False
    if QA_HEADER_RE.match(stripped):
        return False
    return True


def is_speaker_triplet(lines: list[str], index: int) -> bool:
    if index + 2 >= len(lines):
        return False
    return (
        looks_like_speaker_line(lines[index])
        and lines[index + 1].strip() == "--"
        and bool(lines[index + 2].strip())
    )


def find_qa_start(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if QA_HEADER_RE.match(line.strip()):
            return idx

    floor = max(10, math.floor(len(lines) * 0.2))
    for idx in range(floor, len(lines)):
        if QA_TRANSITION_RE.search(lines[idx]):
            return idx
    return None


def parse_blocks(text: str) -> list[Block]:
    lines = text.splitlines()
    qa_start = find_qa_start(lines)
    blocks: list[Block] = []
    section = "prepared"
    idx = 0

    while idx < len(lines):
        if qa_start is not None and idx == qa_start:
            section = "qa"
            if QA_HEADER_RE.match(lines[idx].strip()):
                idx += 1
                continue
            qa_start = None

        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        if QA_HEADER_RE.match(line):
            section = "qa"
            idx += 1
            continue

        if is_speaker_triplet(lines, idx):
            speaker = normalize_space(lines[idx])
            title = normalize_space(lines[idx + 2])
            idx += 3
            body: list[str] = []
            while idx < len(lines):
                if qa_start is not None and idx == qa_start:
                    break
                if QA_HEADER_RE.match(lines[idx].strip()) or is_speaker_triplet(lines, idx):
                    break
                stripped = normalize_space(lines[idx])
                if stripped:
                    body.append(stripped)
                idx += 1
            blocks.append(Block(section=section, speaker=speaker, title=title, text=normalize_space(" ".join(body))))
            continue

        body: list[str] = []
        while idx < len(lines):
            if qa_start is not None and idx == qa_start:
                break
            if QA_HEADER_RE.match(lines[idx].strip()) or is_speaker_triplet(lines, idx):
                break
            stripped = normalize_space(lines[idx])
            if stripped:
                body.append(stripped)
            idx += 1
        joined = normalize_space(" ".join(body))
        if joined:
            blocks.append(Block(section=section, speaker=None, title=None, text=joined))

    return blocks


def normalize_speaker_name(name: str | None) -> str | None:
    if not name:
        return None
    lowered = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", lowered)


def classify_title(title: str | None) -> str:
    lowered = (title or "").lower()
    if "operator" in lowered:
        return "operator"
    if any(hint in lowered for hint in ANALYST_TITLE_HINTS):
        return "analyst"
    if any(hint in lowered for hint in MANAGEMENT_TITLE_HINTS):
        return "management"
    return "unknown"


def classify_block_roles(blocks: list[Block]) -> list[tuple[Block, str]]:
    management_speakers: set[str] = set()
    for block in blocks:
        speaker_key = normalize_speaker_name(block.speaker)
        if block.section != "prepared" or not speaker_key:
            continue
        if speaker_key == "operator":
            continue
        title_role = classify_title(block.title)
        if title_role != "analyst":
            management_speakers.add(speaker_key)

    classified: list[tuple[Block, str]] = []
    for block in blocks:
        speaker_key = normalize_speaker_name(block.speaker)
        title_role = classify_title(block.title)
        if speaker_key == "operator" or title_role == "operator":
            role = "operator"
        elif speaker_key and speaker_key in management_speakers:
            role = "management"
        elif title_role == "management":
            role = "management"
        elif title_role == "analyst":
            role = "analyst"
        elif block.section == "prepared":
            role = "management" if speaker_key else "other"
        elif block.section == "qa":
            role = "analyst" if speaker_key else "other"
        else:
            role = "other"
        classified.append((block, role))
    return classified


def clean_signal_text(text: str) -> str:
    cleaned_sentences: list[str] = []
    for sentence in split_sentences(normalize_space(text)):
        if any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS):
            continue
        cleaned_sentences.append(sentence)
    return normalize_space(" ".join(cleaned_sentences))


def section_texts(text: str) -> dict[str, str]:
    blocks = parse_blocks(text)
    classified = classify_block_roles(blocks)

    prepared_parts: list[str] = []
    qa_parts: list[str] = []
    management_qa_parts: list[str] = []
    analyst_qa_parts: list[str] = []

    for block, role in classified:
        if role == "operator":
            continue
        if block.section == "prepared":
            prepared_parts.append(block.text)
        elif block.section == "qa":
            qa_parts.append(block.text)
            if role == "management":
                management_qa_parts.append(block.text)
            elif role == "analyst":
                analyst_qa_parts.append(block.text)

    prepared = clean_signal_text(" ".join(prepared_parts))
    qa = clean_signal_text(" ".join(qa_parts))
    management_qa = clean_signal_text(" ".join(management_qa_parts))
    analyst_qa = clean_signal_text(" ".join(analyst_qa_parts))
    overall = clean_signal_text(" ".join(part for part in (prepared, qa) if part))

    return {
        "overall": overall,
        "prepared": prepared,
        "qa": qa,
        "management_qa": management_qa,
        "analyst_qa": analyst_qa,
    }


def count_numeric_tokens(tokens: Iterable[str]) -> int:
    return sum(any(character.isdigit() for character in token) for token in tokens)


def complexity_metrics(text: str) -> dict[str, float]:
    sentences = split_sentences(text)
    tokens = tokenize(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    long_word_count = sum(len(token) >= 8 for token in alpha_tokens)
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0.0
    long_word_share = long_word_count / len(alpha_tokens) if alpha_tokens else 0.0
    return {
        "sentence_count": len(sentences),
        "token_count": len(tokens),
        "avg_sentence_length": avg_sentence_length,
        "long_word_share": long_word_share,
        "complexity_score": avg_sentence_length * long_word_share,
        "numeric_token_share": count_numeric_tokens(tokens) / len(tokens) if tokens else 0.0,
    }


def tone_metrics(text: str, prefix: str) -> dict[str, float]:
    token_count = len(tokenize(text))
    positive = COUNTERS["base_positive"].count(text)
    negative = COUNTERS["base_negative"].count(text)
    return {
        f"{prefix}_positive_density": safe_density(positive, token_count),
        f"{prefix}_negative_density": safe_density(negative, token_count),
        f"{prefix}_net_tone": safe_net(positive, negative, token_count),
        f"{prefix}_token_count": token_count,
    }


def topic_metrics(text: str) -> dict[str, float]:
    token_count = len(tokenize(text))
    demand_positive = COUNTERS["demand_positive"].count(text)
    demand_negative = COUNTERS["demand_negative"].count(text)
    pricing_positive = COUNTERS["pricing_positive"].count(text)
    pricing_negative = COUNTERS["pricing_negative"].count(text)
    capex_positive = COUNTERS["capex_positive"].count(text)
    capex_negative = COUNTERS["capex_negative"].count(text)
    supply_pressure = COUNTERS["supply_pressure"].count(text)
    labor_pressure = COUNTERS["labor_pressure"].count(text)
    automation = COUNTERS["automation"].count(text)
    uncertainty = COUNTERS["uncertainty"].count(text)
    risk = COUNTERS["risk"].count(text)
    macro_risk = COUNTERS["macro_risk"].count(text)

    return {
        "demand_positive_density": safe_density(demand_positive, token_count),
        "demand_negative_density": safe_density(demand_negative, token_count),
        "demand_net": safe_net(demand_positive, demand_negative, token_count),
        "pricing_positive_density": safe_density(pricing_positive, token_count),
        "pricing_negative_density": safe_density(pricing_negative, token_count),
        "pricing_power_net": safe_net(pricing_positive, pricing_negative, token_count),
        "capex_positive_density": safe_density(capex_positive, token_count),
        "capex_negative_density": safe_density(capex_negative, token_count),
        "capex_net": safe_net(capex_positive, capex_negative, token_count),
        "supply_chain_pressure_density": safe_density(supply_pressure, token_count),
        "labor_pressure_density": safe_density(labor_pressure, token_count),
        "automation_density": safe_density(automation, token_count),
        "uncertainty_density": safe_density(uncertainty, token_count),
        "risk_density": safe_density(risk, token_count),
        "macro_risk_density": safe_density(macro_risk, token_count),
    }


def guidance_flags(text: str) -> dict[str, int]:
    raised = int(any(pattern.search(text) for pattern in GUIDANCE_RAISED_PATTERNS))
    lowered = int(any(pattern.search(text) for pattern in GUIDANCE_LOWERED_PATTERNS))
    return {
        "guidance_raised": raised,
        "guidance_lowered": lowered,
        "guidance_net": raised - lowered,
    }


def relative_transcript_paths(
    transcripts_dir: Path,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    needle = pattern.lower() if pattern else None
    paths: list[Path] = []
    for path in sorted(transcripts_dir.rglob("*.txt")):
        rel = path.relative_to(transcripts_dir).as_posix().lower()
        if needle and needle not in rel:
            continue
        paths.append(path)
        if limit is not None and len(paths) >= limit:
            break
    return paths


def metadata_for_path(path: Path, transcripts_dir: Path) -> dict[str, str]:
    relative_path = path.relative_to(transcripts_dir)
    match = DATE_IN_STEM_RE.match(path.stem)
    ticker = match.group("ticker") if match else path.stem.split("_")[0]
    call_date = match.group("call_date") if match else ""
    fiscal_period = match.group("fiscal_period") if match else (relative_path.parts[2] if len(relative_path.parts) > 2 else "")
    fiscal_year = relative_path.parts[1] if len(relative_path.parts) > 1 else ""
    company_slug = relative_path.parts[0] if relative_path.parts else path.parent.name
    company_key = ticker if ticker and ticker != "UNKNOWN" else company_slug
    return {
        "transcript_id": relative_path.as_posix().removesuffix(path.suffix),
        "relative_path": relative_path.as_posix(),
        "transcript_path": str(path.resolve()),
        "company_slug": company_slug,
        "company_name": slug_to_name(company_slug),
        "ticker": ticker,
        "company_key": company_key.upper(),
        "call_date": call_date,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "file_name": path.name,
    }


def score_transcript(path: Path, transcripts_dir: Path) -> dict[str, object]:
    metadata = metadata_for_path(path, transcripts_dir)
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    sections = section_texts(raw_text)

    overall_text = sections["overall"]
    prepared_text = sections["prepared"]
    qa_text = sections["qa"]
    management_qa_text = sections["management_qa"]
    analyst_qa_text = sections["analyst_qa"]

    overall_complexity = complexity_metrics(overall_text)
    prepared_tone = tone_metrics(prepared_text, "prepared")
    qa_tone = tone_metrics(qa_text, "qa")
    management_qa_tone = tone_metrics(management_qa_text, "management_qa")
    analyst_qa_tone = tone_metrics(analyst_qa_text, "analyst_question")
    overall_tone = tone_metrics(overall_text, "overall")
    topical = topic_metrics(overall_text)
    guidance = guidance_flags(raw_text)

    qna_reality_gap = management_qa_tone["management_qa_net_tone"] - prepared_tone["prepared_net_tone"]
    analyst_management_gap = management_qa_tone["management_qa_net_tone"] - analyst_qa_tone["analyst_question_net_tone"]
    growth_signal = overall_tone["overall_net_tone"] + topical["demand_net"] + topical["capex_net"] - topical["macro_risk_density"]
    margin_signal = (
        topical["pricing_power_net"]
        + topical["automation_density"]
        - topical["labor_pressure_density"]
        - topical["supply_chain_pressure_density"]
    )
    credibility_signal = analyst_management_gap + qna_reality_gap - topical["uncertainty_density"]
    composite_signal = growth_signal + margin_signal + credibility_signal

    return {
        **metadata,
        "qa_detected": int(bool(qa_text)),
        **overall_complexity,
        **overall_tone,
        **prepared_tone,
        **qa_tone,
        **management_qa_tone,
        **analyst_qa_tone,
        **topical,
        **guidance,
        "qna_reality_gap": qna_reality_gap,
        "analyst_management_gap": analyst_management_gap,
        "growth_signal": growth_signal,
        "margin_signal": margin_signal,
        "credibility_signal": credibility_signal,
        "composite_signal": composite_signal,
    }


def score_transcript_worker(path_str: str, transcripts_dir_str: str) -> dict[str, object]:
    return score_transcript(Path(path_str), Path(transcripts_dir_str))


def existing_fieldnames(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, None)


def load_processed_ids(raw_path: Path) -> set[str]:
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return set()
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["transcript_id"] for row in reader if row.get("transcript_id")}


def open_append_writer(raw_path: Path, fieldnames: list[str]) -> tuple[object, csv.DictWriter]:
    write_header = not raw_path.exists() or raw_path.stat().st_size == 0
    handle = raw_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def append_raw_row(writer: csv.DictWriter, handle, fieldnames: list[str], row: dict[str, object]) -> None:
    writer.writerow({field: row.get(field) for field in fieldnames})
    handle.flush()


def iter_scored_rows(
    paths: list[Path],
    transcripts_dir: Path,
    workers: int,
) -> Iterable[dict[str, object]]:
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(
                    score_transcript_worker,
                    (str(path) for path in paths),
                    repeat(str(transcripts_dir)),
                )
                for row in tqdm(iterator, total=len(paths), desc="Scoring transcripts"):
                    yield row
            return
        except (OSError, PermissionError) as error:
            LOGGER.warning("Falling back to single-process scoring: %s", error)

    for path in tqdm(paths, desc="Scoring transcripts"):
        yield score_transcript(path, transcripts_dir)


def prepare_output_files(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    paths = {
        "raw": output_dir / "earnings_research_signal_raw.csv",
        "panel": output_dir / "earnings_research_signal_panel.csv",
        "monthly": output_dir / "earnings_research_macro_monthly.csv",
        "quarterly": output_dir / "earnings_research_macro_quarterly.csv",
        "latest": output_dir / "earnings_research_latest_snapshot.csv",
    }
    if overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()
    return paths


def dedupe_panel(panel: pd.DataFrame) -> pd.DataFrame:
    dedupe_date = panel["call_date"].fillna("").astype(str)
    dedupe_period = panel["fiscal_period"].fillna("").astype(str)
    panel = panel.assign(dedupe_key=panel["company_key"] + "|" + dedupe_date + "|" + dedupe_period)
    duplicate_counts = panel["dedupe_key"].value_counts().rename("duplicates_in_group")
    panel = panel.merge(duplicate_counts, left_on="dedupe_key", right_index=True, how="left")
    panel = panel.sort_values(
        by=["company_key", "call_date_dt", "fiscal_period", "token_count"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    panel = panel.drop_duplicates(subset=["dedupe_key"], keep="first").copy()
    return panel.drop(columns=["dedupe_key"])


def add_company_time_series_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(by=["company_key", "call_date_dt", "fiscal_year", "fiscal_period"]).copy()
    for column in ROLLING_SIGNAL_COLUMNS:
        grouped = panel.groupby("company_key")[column]
        panel[f"{column}_qoq_change"] = grouped.diff()
        panel[f"{column}_yoy_change"] = grouped.diff(4)
        panel[f"{column}_rolling_z8"] = grouped.transform(rolling_zscore)
        panel[f"{column}_rolling_mean4"] = grouped.transform(lambda series: series.rolling(4, min_periods=2).mean())

    if "call_month" in panel.columns:
        panel["month_composite_rank_pct"] = panel.groupby("call_month")["composite_signal"].rank(pct=True)
    else:
        panel["month_composite_rank_pct"] = np.nan

    if "call_quarter" in panel.columns:
        panel["quarter_composite_rank_pct"] = panel.groupby("call_quarter")["composite_signal"].rank(pct=True)
    else:
        panel["quarter_composite_rank_pct"] = np.nan

    return panel


def weighted_average(frame: pd.DataFrame, column: str, weight_column: str = "token_count") -> float:
    weights = frame[weight_column].fillna(0).astype(float)
    values = frame[column].fillna(0).astype(float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float(values.mean()) if not values.empty else 0.0
    return float(np.average(values, weights=weights))


def aggregate_panel(panel: pd.DataFrame, period_column: str) -> pd.DataFrame:
    valid = panel.dropna(subset=[period_column]).copy()
    rows: list[dict[str, object]] = []
    for period_value, frame in valid.groupby(period_column):
        row: dict[str, object] = {
            period_column: period_value,
            "n_transcripts": int(len(frame)),
            "n_companies": int(frame["company_key"].nunique()),
            "share_guidance_raised": float(frame["guidance_raised"].mean()),
            "share_guidance_lowered": float(frame["guidance_lowered"].mean()),
            "share_positive_composite": float((frame["composite_signal"] > 0).mean()),
            "share_negative_qna_gap": float((frame["qna_reality_gap"] < 0).mean()),
        }
        for column in AGGREGATION_COLUMNS:
            row[f"ew_{column}"] = float(frame[column].mean())
            row[f"tw_{column}"] = weighted_average(frame, column)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(by=period_column).reset_index(drop=True)
    for column in ("ew_composite_signal", "ew_growth_signal", "ew_margin_signal", "ew_macro_risk_density"):
        if column in result.columns:
            result[f"{column}_rolling_mean4"] = result[column].rolling(4, min_periods=2).mean()
            result[f"{column}_rolling_z8"] = rolling_zscore(result[column])
    return result


def build_latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    snapshot = panel.sort_values(by=["company_key", "call_date_dt", "fiscal_year", "fiscal_period"]).copy()
    snapshot = snapshot.drop_duplicates(subset=["company_key"], keep="last")
    sort_columns = [column for column in ("composite_signal_rolling_z8", "composite_signal", "growth_signal") if column in snapshot]
    if sort_columns:
        snapshot = snapshot.sort_values(by=sort_columns, ascending=False)
    return snapshot.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.transcripts_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {args.transcripts_dir}")

    ensure_dir(args.output_dir)
    output_paths = prepare_output_files(args.output_dir, overwrite=args.overwrite)

    paths = relative_transcript_paths(args.transcripts_dir, pattern=args.pattern, limit=args.limit)
    if not paths:
        raise FileNotFoundError("No transcript files matched the requested filters.")

    processed_ids = load_processed_ids(output_paths["raw"])
    if processed_ids:
        LOGGER.warning("Found %s raw rows already written; resume mode is active", len(processed_ids))
        paths = [
            path
            for path in paths
            if metadata_for_path(path, args.transcripts_dir)["transcript_id"] not in processed_ids
        ]

    starting_raw_count = len(processed_ids)
    total_target_count = starting_raw_count + len(paths)
    LOGGER.warning(
        "Scoring %s remaining transcript files with a lexicon-only pipeline (%s total target rows)",
        len(paths),
        total_target_count,
    )
    if not paths:
        LOGGER.warning("No new transcript rows need scoring; rebuilding final outputs from the existing raw CSV")
    fieldnames = existing_fieldnames(output_paths["raw"])
    writer_handle = None
    writer = None
    appended_this_run = 0
    run_start = time.perf_counter()
    try:
        for row in iter_scored_rows(paths, args.transcripts_dir, workers=args.workers):
            if fieldnames is None:
                fieldnames = list(row.keys())
                writer_handle, writer = open_append_writer(output_paths["raw"], fieldnames)
            elif writer is None:
                row_fieldnames = list(row.keys())
                if fieldnames != row_fieldnames:
                    raise ValueError(
                        "Existing raw CSV schema does not match the current script output. "
                        "Re-run with --overwrite to rebuild the output directory."
                    )
                writer_handle, writer = open_append_writer(output_paths["raw"], fieldnames)
            append_raw_row(writer, writer_handle, fieldnames, row)
            appended_this_run += 1
            should_log = (
                appended_this_run == 1
                or appended_this_run == len(paths)
                or appended_this_run % max(1, args.log_every) == 0
            )
            if should_log:
                elapsed = max(time.perf_counter() - run_start, 1e-9)
                rate = appended_this_run / elapsed
                total_written = starting_raw_count + appended_this_run
                remaining = len(paths) - appended_this_run
                LOGGER.warning(
                    "Appended %s / %s rows to %s (new this run: %s, remaining: %s, rate: %.2f rows/s)",
                    total_written,
                    total_target_count,
                    output_paths["raw"],
                    appended_this_run,
                    remaining,
                    rate,
                )
    finally:
        if writer_handle is not None:
            writer_handle.close()

    if not output_paths["raw"].exists():
        raise FileNotFoundError("No raw transcript rows were written; nothing to aggregate.")

    panel = pd.read_csv(output_paths["raw"])
    panel["call_date_dt"] = pd.to_datetime(panel["call_date"], errors="coerce")
    panel["call_month"] = panel["call_date_dt"].dt.to_period("M").astype("string")
    panel["call_quarter"] = panel["call_date_dt"].dt.to_period("Q").astype("string")

    panel = dedupe_panel(panel)
    panel = add_company_time_series_features(panel)

    monthly = aggregate_panel(panel, "call_month")
    quarterly = aggregate_panel(panel, "call_quarter")
    latest = build_latest_snapshot(panel)

    panel.to_csv(output_paths["panel"], index=False)
    monthly.to_csv(output_paths["monthly"], index=False)
    quarterly.to_csv(output_paths["quarterly"], index=False)
    latest.to_csv(output_paths["latest"], index=False)

    LOGGER.warning("Raw transcript rows are stored at %s", output_paths["raw"])
    LOGGER.warning("Wrote panel to %s", output_paths["panel"])
    LOGGER.warning("Wrote monthly macro rollup to %s", output_paths["monthly"])
    LOGGER.warning("Wrote quarterly macro rollup to %s", output_paths["quarterly"])
    LOGGER.warning("Wrote latest snapshot to %s", output_paths["latest"])


if __name__ == "__main__":
    main()
