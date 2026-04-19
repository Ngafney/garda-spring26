import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.policy.policy_score import (
    DOC_TYPE_REGIONAL_WEIGHTS,
    STANDARDIZATION_DENOMINATOR,
    category_adjustment_weight,
    detect_categories,
    is_question_like_sentence,
    magnitude_score,
    sentiment_score,
    split_sentences as split_policy_sentences,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent
WORD_RE = re.compile(r"\b[\w%./-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class LexiconSpec:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


@dataclass(frozen=True)
class BucketSpec:
    slug: str
    label: str
    components: tuple[str, ...]
    color: str


class PhraseCounter:
    def __init__(self, phrases: tuple[str, ...]):
        escaped = [re.escape(phrase.lower()) for phrase in phrases if phrase]
        self.pattern = (
            re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), flags=re.IGNORECASE)
            if escaped
            else None
        )

    def count(self, text: str) -> int:
        return len(self.pattern.findall(text)) if self.pattern and text else 0


DEFAULT_WORD_BANK: dict[str, LexiconSpec] = {
    "base_sentiment": LexiconSpec(
        positive=(
            "accelerating", "accretive", "benefit", "confidence", "constructive", "durable",
            "efficient", "expand", "favorable", "growth", "healthy", "improve", "momentum",
            "opportunity", "positive", "productivity", "record", "resilient", "robust",
            "solid", "stable", "strength", "strong", "upside",
        ),
        negative=(
            "challenging", "constraint", "cautious", "decline", "difficult", "downturn",
            "erosion", "headwind", "loss", "negative", "pressure", "recession", "risk",
            "slowdown", "soft", "uncertain", "uncertainty", "volatility", "weak", "weakness",
        ),
    ),
    "uncertainty": LexiconSpec(
        positive=("visibility", "clearer outlook", "predictable", "stabilizing", "certainty", "transparency"),
        negative=("uncertain", "uncertainty", "volatile", "limited visibility", "not clear", "unknown", "cautious"),
    ),
    "demand": LexiconSpec(
        positive=(
            "strong demand", "healthy demand", "robust demand", "solid demand", "better demand",
            "improving demand", "stable demand", "order growth", "bookings growth", "backlog growth",
            "good demand", "demand recovery", "volume growth", "share gains", "market share gains",
        ),
        negative=(
            "weak demand", "soft demand", "demand slowdown", "slowing demand", "lower demand",
            "demand pressure", "order weakness", "bookings weakness", "backlog pressure",
            "customer caution", "cautious customer", "destocking", "inventory correction",
            "volume pressure", "traffic weakness",
        ),
    ),
    "pricing": LexiconSpec(
        positive=(
            "pricing power", "price increase", "price increases", "positive pricing", "favorable pricing",
            "pricing discipline", "price realization", "net price", "margin expansion", "mix benefit",
            "premiumization", "higher price", "pass-through", "pass through", "pricing actions",
        ),
        negative=(
            "price pressure", "pricing pressure", "promotional", "promotions", "discounting", "discounts",
            "margin pressure", "cost inflation", "inflationary pressure", "input cost",
            "commodity inflation", "mix headwind", "unfavorable mix", "deflation", "price elasticity",
        ),
    ),
    "supply_pressure": LexiconSpec(
        positive=(
            "supply easing", "normalization", "inventory health", "improved lead times",
            "logistics recovery", "de-bottlenecking", "resolved shortages",
        ),
        negative=(
            "supply chain", "shortage", "shortages", "constraint", "constraints", "constrained",
            "bottleneck", "bottlenecks", "lead time", "lead times", "backorder", "backorders",
            "shipping delay", "freight pressure", "logistics pressure", "inventory shortage",
            "component shortage",
        ),
    ),
    "capex": LexiconSpec(
        positive=(
            "capital expenditure", "capex", "investment", "investing", "capacity expansion", "buildout",
            "factory expansion", "new plant", "greenfield", "brownfield", "data center",
            "expansion project", "ramping capacity", "automation investment", "infrastructure investment",
        ),
        negative=(
            "cut capex", "reduce capex", "lower capex", "pause investment", "delay investment",
            "project delay", "project delays", "cancel project", "capacity reduction",
        ),
    ),
    "gdp_growth": LexiconSpec(
        positive=(
            "gdp growth", "economic growth", "growth outlook", "growth recovery", "soft landing",
            "resilient economy", "economic momentum", "above-trend growth", "trend growth",
        ),
        negative=(
            "economic contraction", "growth slowdown", "slowing growth", "weak growth",
            "stagnation", "recession", "hard landing", "economic weakness", "macro slowdown",
        ),
    ),
    "ai": LexiconSpec(
        positive=("ai", "artificial intelligence", "machine learning", "automation", "productivity gains", "efficiency gains"),
        negative=("ai risk", "ai regulation", "automation risk", "compute constraint", "ai bubble"),
    ),
    "unemployment": LexiconSpec(
        positive=(
            "low unemployment", "job growth", "employment growth", "strong labor market",
            "healthy labor market", "labor market strength", "solid hiring", "wage growth",
        ),
        negative=(
            "high unemployment", "unemployment increased", "job losses", "employment decline",
            "labor market weakness", "weak labor market", "slowing hiring", "layoffs",
            "job cuts", "rising unemployment",
        ),
    ),
    "inflation": LexiconSpec(
        positive=(
            "disinflation", "inflation easing", "lower inflation", "inflation moderated",
            "cooling inflation", "price stability", "stable prices", "cost deflation",
        ),
        negative=(
            "inflation", "inflationary", "high inflation", "higher inflation", "sticky inflation",
            "persistent inflation", "inflation pressure", "cost inflation", "wage inflation",
            "price increases",
        ),
    ),
    "rates": LexiconSpec(
        positive=(
            "rate cut", "rate cuts", "lower rates", "policy easing", "easing cycle",
            "accommodative", "reduce restraint", "less restrictive", "lowered the policy rate",
        ),
        negative=(
            "rate hike", "rate hikes", "higher rates", "raise rates", "policy tightening",
            "tightening cycle", "restrictive", "sufficiently restrictive", "higher for longer",
        ),
    ),
    "financial_conditions": LexiconSpec(
        positive=(
            "easing financial conditions", "improved financial conditions", "ample liquidity",
            "orderly market functioning", "lower borrowing costs", "narrower spreads", "stable funding markets",
        ),
        negative=(
            "tight financial conditions", "tighter financial conditions", "credit tightening",
            "funding stress", "market dysfunction", "elevated spreads", "banking stress",
            "financial stability risk", "liquidity strains", "credit stress",
        ),
    ),
}

COMMON_BUCKETS = (
    BucketSpec("demand", "Demand", ("demand_net", "gdp_growth_net"), "#4C72B0"),
    BucketSpec("inflation", "Inflation", ("inflation_net", "supply_pressure_net", "pricing_net"), "#C44E52"),
    BucketSpec("employment", "Employment", ("unemployment_net",), "#8172B2"),
    BucketSpec("capex", "Capex", ("capex_net",), "#DD8452"),
    BucketSpec("ai", "AI", ("ai_net",), "#55A868"),
)
CENTRAL_BANK_BUCKETS = COMMON_BUCKETS + (
    BucketSpec("hawkish_dovish", "Hawkish / Dovish", ("hawkish_dovish_net",), "#937860"),
)
EARNINGS_BUCKETS = (
    BucketSpec("demand", "Demand", ("demand_net", "gdp_growth_net"), "#4C72B0"),
    BucketSpec("inflation", "Inflation", ("inflation_net", "supply_chain_pressure_density", "pricing_power_net"), "#C44E52"),
    BucketSpec("employment", "Employment", ("unemployment_net", "labor_pressure_density"), "#8172B2"),
    BucketSpec("capex", "Capex", ("capex_net",), "#DD8452"),
    BucketSpec("ai", "AI", ("ai_net", "automation_density"), "#55A868"),
)
TOPIC_COMPONENTS = (
    "demand",
    "pricing",
    "supply_pressure",
    "capex",
    "inflation",
    "gdp_growth",
    "ai",
    "unemployment",
    "uncertainty",
    "rates",
    "financial_conditions",
)


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def split_terms(cell: object) -> tuple[str, ...]:
    if pd.isna(cell):
        return ()
    return tuple(part.strip() for part in str(cell).split(",") if part.strip())


def load_word_bank(workbook_path: Path) -> dict[str, LexiconSpec]:
    lexicons = dict(DEFAULT_WORD_BANK)
    if not workbook_path.exists():
        LOGGER.warning("Workbook %s not found; using built-in lexicons.", workbook_path)
        return lexicons
    try:
        df = pd.read_excel(workbook_path, sheet_name="Lexicons")
    except Exception as exc:
        LOGGER.warning("Could not read %s (%s); using built-in lexicons.", workbook_path, exc)
        return lexicons
    for _, row in df.iterrows():
        key = normalize_key(row.get("Category", ""))
        if not key:
            continue
        lexicons[key] = LexiconSpec(
            positive=split_terms(row.get("Positive Terms")),
            negative=split_terms(row.get("Negative Terms")),
        )
    return lexicons


def build_counters(lexicons: dict[str, LexiconSpec]) -> dict[str, PhraseCounter]:
    counters = {
        "base_positive": PhraseCounter(lexicons["base_sentiment"].positive),
        "base_negative": PhraseCounter(lexicons["base_sentiment"].negative),
    }
    for name in TOPIC_COMPONENTS:
        counters[f"{name}_positive"] = PhraseCounter(lexicons[name].positive)
        counters[f"{name}_negative"] = PhraseCounter(lexicons[name].negative)
    return counters


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_basic_sentences(text: str) -> list[str]:
    return [normalize_space(part) for part in SENTENCE_RE.split(text) if normalize_space(part)] if text else []


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def safe_net(pos: int, neg: int, denom: int) -> float:
    return (pos - neg) / denom if denom > 0 else 0.0


def safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def safe_mean_row(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[available].astype(float).mean(axis=1, skipna=True)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def clean_text(text: str) -> str:
    return normalize_space(" ".join(split_basic_sentences(text)))


def complexity_metrics(text: str) -> dict[str, float]:
    sentences = split_basic_sentences(text)
    tokens = tokenize(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0.0
    long_word_share = sum(len(token) >= 8 for token in alpha_tokens) / len(alpha_tokens) if alpha_tokens else 0.0
    return {
        "sentence_count": len(sentences),
        "token_count": len(tokens),
        "avg_sentence_length": avg_sentence_length,
        "long_word_share": long_word_share,
        "complexity_score": avg_sentence_length * long_word_share,
    }


def hawkish_dovish_score(raw_text: str, doc_type: str) -> float:
    weighted_scores: list[float] = []
    last_categories: list[str] = []
    doc_type_key = normalize_key(doc_type)

    for sentence in split_policy_sentences(raw_text):
        if doc_type_key == "press_conference" and is_question_like_sentence(sentence):
            continue
        categories = detect_categories(sentence)
        if not categories and last_categories:
            referential_openers = ("this ", "these ", "it ", "they ", "such ", "those ")
            if sentence.lower().startswith(referential_openers):
                categories = last_categories.copy()
        if categories:
            last_categories = categories.copy()
        if not categories:
            continue
        sentiment = sentiment_score(sentence, categories)
        if sentiment == 0:
            continue
        weighted_scores.append(sentiment * magnitude_score(sentence) * category_adjustment_weight(categories))

    if not weighted_scores:
        return 0.0

    avg_document_signal = sum(weighted_scores) / len(weighted_scores)
    doc_type_weight = DOC_TYPE_REGIONAL_WEIGHTS.get(doc_type_key, 1.0)
    # Flip the sign so positive means more hawkish and negative means more dovish.
    scaled = (-avg_document_signal / STANDARDIZATION_DENOMINATOR) * doc_type_weight
    return float(np.clip(scaled, -1.0, 1.0))


def score_topic_terms(text: str, raw_text: str, counters: dict[str, PhraseCounter], doc_type: str) -> dict[str, float]:
    token_count = len(tokenize(text))
    metrics = {}
    for name in TOPIC_COMPONENTS:
        pos = counters[f"{name}_positive"].count(text)
        neg = counters[f"{name}_negative"].count(text)
        metrics[f"{name}_net"] = safe_net(pos, neg, token_count)
    pos = counters["base_positive"].count(text)
    neg = counters["base_negative"].count(text)
    metrics["overall_net_tone"] = safe_net(pos, neg, token_count)
    metrics["hawkish_dovish_net"] = hawkish_dovish_score(raw_text, doc_type)
    return metrics


def parse_year_quarter(date_text: str) -> tuple[int | None, int | None, str | None]:
    dt = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(dt):
        return None, None, None
    return int(dt.year), int(dt.quarter), f"{dt.year}-Q{dt.quarter}"


def build_metadata_map(metadata_path: Path | None) -> dict[str, dict[str, object]]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    df = pd.read_csv(metadata_path)
    df["text_file"] = df["text_file"].astype(str).str.replace("\\", "/", regex=False)
    return {row["text_file"]: row.to_dict() for _, row in df.iterrows()}


def score_policy_document(path: Path, root: Path, metadata_map: dict[str, dict[str, object]], counters: dict[str, PhraseCounter]) -> dict[str, object]:
    rel_path = str(path.relative_to(root)).replace("\\", "/")
    meta = metadata_map.get(rel_path, {})
    parts = path.relative_to(root).parts
    doc_type = str(meta.get("doc_type") or (parts[1] if len(parts) > 1 else ""))

    raw_text = read_text(path)
    text = clean_text(raw_text)
    metrics = complexity_metrics(text)
    metrics.update(score_topic_terms(text, raw_text, counters, doc_type))

    date_text = str(meta.get("date") or "")
    if not date_text:
        match = DATE_RE.search(path.name)
        date_text = match.group(1) if match else ""
    period_year, period_quarter, year_quarter = parse_year_quarter(date_text)

    return {
        "source_file": rel_path,
        "central_bank": str(meta.get("central_bank") or (parts[0].upper() if parts else "UNKNOWN")),
        "region": str(meta.get("region") or ""),
        "doc_type": doc_type,
        "speaker": str(meta.get("speaker") or ""),
        "date": date_text,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "year_quarter": year_quarter,
        **metrics,
    }


def add_bucket_columns(
    df: pd.DataFrame,
    bucket_specs: tuple[BucketSpec, ...],
    group_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    scored = df.copy()
    for bucket in bucket_specs:
        raw_col = f"{bucket.slug}_raw"
        z_col = f"{bucket.slug}_zscore"
        scored[raw_col] = safe_mean_row(scored, bucket.components)
        valid = scored[raw_col].notna()
        scored[z_col] = 0.0
        if not valid.any():
            continue
        if group_cols:
            scored.loc[valid, z_col] = (
                scored.loc[valid]
                .groupby(list(group_cols), dropna=False)[raw_col]
                .transform(safe_zscore)
                .fillna(0.0)
                .values
            )
        else:
            scored.loc[valid, z_col] = safe_zscore(scored.loc[valid, raw_col]).values
    return scored


def bucket_zscore_long(df: pd.DataFrame, bucket_specs: tuple[BucketSpec, ...], period_label_col: str, source_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        for bucket in bucket_specs:
            rows.append({
                "source": source_name,
                "period_year": row["period_year"],
                "period_quarter": row["period_quarter"],
                "period_label": row[period_label_col],
                "bucket": bucket.label,
                "bucket_slug": bucket.slug,
                "raw_score": row[f"{bucket.slug}_raw"],
                "z_score": row[f"{bucket.slug}_zscore"],
            })
    return pd.DataFrame(rows)


def plot_bucket_series(
    quarterly: pd.DataFrame,
    bucket: BucketSpec,
    period_label_col: str,
    source_title: str,
    x_label: str,
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.plot(
        quarterly[period_label_col],
        quarterly[f"{bucket.slug}_zscore"],
        marker="o",
        linewidth=2.35,
        markersize=4.8,
        color=bucket.color,
    )
    ax.axhline(0, color="#444444", linewidth=1, alpha=0.75)
    ax.set_title(f"{source_title} {bucket.label} Signal Over Time", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel(x_label, fontsize=11)
    if bucket.slug == "hawkish_dovish":
        ax.set_ylabel("Z-Score vs. source history\n(positive = more hawkish)", fontsize=11)
    else:
        ax.set_ylabel("Z-Score vs. source history", fontsize=11)
    ax.tick_params(axis="x", rotation=50)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bucket_plots(
    quarterly: pd.DataFrame,
    bucket_specs: tuple[BucketSpec, ...],
    period_label_col: str,
    output_dir: Path,
    source_title: str,
    file_prefix: str,
    x_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket in bucket_specs:
        plot_bucket_series(
            quarterly=quarterly,
            bucket=bucket,
            period_label_col=period_label_col,
            source_title=source_title,
            x_label=x_label,
            output_path=output_dir / f"{file_prefix}_{bucket.slug}_signal_over_time.png",
        )


def build_group_display_name(group: pd.DataFrame, group_value: object) -> str:
    label = str(group_value)
    if "region" not in group.columns:
        return label
    regions = [str(value).strip() for value in group["region"].dropna().unique() if str(value).strip()]
    if not regions:
        return label
    region = regions[0]
    return label if region.lower() in label.lower() else f"{label} ({region})"


def save_grouped_bucket_plots(
    frame: pd.DataFrame,
    group_col: str,
    bucket_specs: tuple[BucketSpec, ...],
    period_label_col: str,
    output_dir: Path,
    file_prefix: str,
    x_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_value, group in frame.groupby(group_col, dropna=False):
        group = group.sort_values(["period_year", "period_quarter"]).reset_index(drop=True)
        display_name = build_group_display_name(group, group_value)
        group_dir = output_dir / normalize_key(display_name)
        group_dir.mkdir(parents=True, exist_ok=True)
        for bucket in bucket_specs:
            plot_bucket_series(
                quarterly=group,
                bucket=bucket,
                period_label_col=period_label_col,
                source_title=display_name,
                x_label=x_label,
                output_path=group_dir / f"{file_prefix}_{bucket.slug}_signal_over_time.png",
            )


def load_earnings_bucket_quarterly(earnings_panel_path: Path, earnings_category_wide_path: Path) -> pd.DataFrame | None:
    if not earnings_panel_path.exists():
        LOGGER.warning("Earnings panel %s not found; skipping earnings plots.", earnings_panel_path)
        return None
    if not earnings_category_wide_path.exists():
        LOGGER.warning("Earnings category file %s not found; skipping earnings plots.", earnings_category_wide_path)
        return None

    earnings_panel = pd.read_csv(earnings_panel_path)
    earnings_panel["call_date_dt"] = pd.to_datetime(earnings_panel.get("call_date_dt", earnings_panel.get("call_date")), errors="coerce")
    earnings_panel = earnings_panel.dropna(subset=["call_date_dt"]).copy()
    earnings_panel["period_year"] = earnings_panel["call_date_dt"].dt.year.astype(int)
    earnings_panel["period_quarter"] = earnings_panel["call_date_dt"].dt.quarter.astype(int)

    panel_component_cols = [
        "demand_net",
        "pricing_power_net",
        "capex_net",
        "supply_chain_pressure_density",
        "labor_pressure_density",
        "automation_density",
    ]
    earnings_panel_quarterly = (
        earnings_panel.groupby(["period_year", "period_quarter"], as_index=False)[panel_component_cols]
        .mean()
        .sort_values(["period_year", "period_quarter"])
    )

    transcript_quarterly = pd.read_csv(earnings_category_wide_path)
    transcript_component_cols = ["period_year", "period_quarter", "inflation_net", "gdp_growth_net", "ai_net", "unemployment_net"]
    transcript_quarterly = transcript_quarterly[transcript_component_cols].copy()

    quarterly = (
        earnings_panel_quarterly.merge(transcript_quarterly, on=["period_year", "period_quarter"], how="outer")
        .sort_values(["period_year", "period_quarter"])
        .reset_index(drop=True)
    )
    quarterly["period_year"] = quarterly["period_year"].astype(int)
    quarterly["period_quarter"] = quarterly["period_quarter"].astype(int)
    quarterly["period_label"] = quarterly["period_year"].astype(str) + "-Q" + quarterly["period_quarter"].astype(str)
    return add_bucket_columns(quarterly, EARNINGS_BUCKETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score central bank policy texts and build bucket plots for central banks and earnings calls.")
    parser.add_argument("--input-root", required=True, help="Root folder containing policy text files.")
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV with central bank/date/doc-type info.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--word-bank", default="word_bank.xlsx", help="Workbook containing lexicon categories.")
    parser.add_argument(
        "--earnings-panel",
        default=str(REPO_ROOT / "earnings_research_signal_panel.csv"),
        help="Quarterly earnings signal panel with demand/pricing/capex-style metrics.",
    )
    parser.add_argument(
        "--earnings-category-wide",
        default=str(REPO_ROOT / "outputs" / "all_transcripts_call_date_category_zscores_over_time_wide.csv"),
        help="Quarterly transcript-category file with inflation/growth/employment/AI metrics for earnings calls.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max((os.cpu_count() or 2) - 1, 1),
        help="Parallel workers for document scoring.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lexicons = load_word_bank(Path(args.word_bank).resolve())
    counters = build_counters(lexicons)
    metadata_map = build_metadata_map(Path(args.metadata).resolve()) if args.metadata else {}

    txt_files = sorted(input_root.rglob("*.txt"))
    print(f"--> SYSTEM FOUND {len(txt_files)} POLICY TEXT FILES <--")
    if not txt_files:
        raise SystemExit("No .txt files found.")

    if args.workers == 1:
        rows = [score_policy_document(path, input_root, metadata_map, counters) for path in tqdm(txt_files, desc="Scoring policy")]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(score_policy_document, path, input_root, metadata_map, counters)
                for path in txt_files
            ]
            rows = [future.result() for future in tqdm(futures, desc="Scoring policy")]

    panel = pd.DataFrame(rows)
    panel = panel.dropna(subset=["period_year", "period_quarter", "year_quarter"]).copy()
    panel["period_year"] = panel["period_year"].astype(int)
    panel["period_quarter"] = panel["period_quarter"].astype(int)

    bank_quarter = (
        panel.groupby(["central_bank", "region", "period_year", "period_quarter", "year_quarter"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["central_bank", "period_year", "period_quarter"])
    )
    bank_quarter = add_bucket_columns(bank_quarter, CENTRAL_BANK_BUCKETS, group_cols=("central_bank",))

    policy_quarterly = (
        panel.groupby(["period_year", "period_quarter", "year_quarter"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["period_year", "period_quarter"])
    )
    policy_quarterly = add_bucket_columns(policy_quarterly, CENTRAL_BANK_BUCKETS)
    policy_quarterly_long = bucket_zscore_long(policy_quarterly, CENTRAL_BANK_BUCKETS, "year_quarter", "central_bank")

    panel.to_csv(output_dir / "policy_research_doc_scores.csv", index=False)
    bank_quarter.to_csv(output_dir / "policy_research_bank_quarterly.csv", index=False)
    policy_quarterly.to_csv(output_dir / "policy_research_bucket_zscores_over_time_wide.csv", index=False)
    policy_quarterly_long.to_csv(output_dir / "policy_research_bucket_zscores_over_time_long.csv", index=False)
    save_bucket_plots(
        quarterly=policy_quarterly,
        bucket_specs=CENTRAL_BANK_BUCKETS,
        period_label_col="year_quarter",
        output_dir=output_dir / "central_bank_bucket_plots",
        source_title="All Central Banks",
        file_prefix="central_bank",
        x_label="Quarter",
    )
    save_grouped_bucket_plots(
        frame=bank_quarter,
        group_col="central_bank",
        bucket_specs=CENTRAL_BANK_BUCKETS,
        period_label_col="year_quarter",
        output_dir=output_dir / "central_bank_bucket_plots_by_bank",
        file_prefix="central_bank",
        x_label="Quarter",
    )

    print(f"Saved {output_dir / 'policy_research_doc_scores.csv'}")
    print(f"Saved {output_dir / 'policy_research_bank_quarterly.csv'}")
    print(f"Saved {output_dir / 'policy_research_bucket_zscores_over_time_wide.csv'}")
    print(f"Saved {output_dir / 'policy_research_bucket_zscores_over_time_long.csv'}")
    print(f"Saved {output_dir / 'central_bank_bucket_plots'}")
    print(f"Saved {output_dir / 'central_bank_bucket_plots_by_bank'}")

    earnings_quarterly = load_earnings_bucket_quarterly(
        earnings_panel_path=Path(args.earnings_panel).resolve(),
        earnings_category_wide_path=Path(args.earnings_category_wide).resolve(),
    )
    if earnings_quarterly is not None and not earnings_quarterly.empty:
        earnings_long = bucket_zscore_long(earnings_quarterly, EARNINGS_BUCKETS, "period_label", "earnings_calls")
        earnings_quarterly.to_csv(output_dir / "earnings_calls_bucket_zscores_over_time_wide.csv", index=False)
        earnings_long.to_csv(output_dir / "earnings_calls_bucket_zscores_over_time_long.csv", index=False)
        save_bucket_plots(
            quarterly=earnings_quarterly,
            bucket_specs=EARNINGS_BUCKETS,
            period_label_col="period_label",
            output_dir=output_dir / "earnings_call_bucket_plots",
            source_title="Earnings Call",
            file_prefix="earnings_call",
            x_label="Quarter",
        )

        print(f"Saved {output_dir / 'earnings_calls_bucket_zscores_over_time_wide.csv'}")
        print(f"Saved {output_dir / 'earnings_calls_bucket_zscores_over_time_long.csv'}")
        print(f"Saved {output_dir / 'earnings_call_bucket_plots'}")


if __name__ == "__main__":
    main()
