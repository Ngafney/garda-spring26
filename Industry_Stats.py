import argparse
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)
WORD_RE = re.compile(r"\b[\w%./-]+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class LexiconSpec:
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


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

POLICY_CATEGORY_COLS = {
    "Inflation": "inflation_net",
    "Growth": "gdp_growth_net",
    "Labor": "unemployment_net",
    "Rates": "rates_net",
    "Financial Conditions": "financial_conditions_net",
    "Uncertainty": "uncertainty_net",
}

PLOT_COLORS = {
    "Inflation": "#C44E52",
    "Growth": "#4C72B0",
    "Labor": "#8172B2",
    "Rates": "#64B5CD",
    "Financial Conditions": "#8C8C8C",
    "Uncertainty": "#CCB974",
}


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
    return {
        "base_positive": PhraseCounter(lexicons["base_sentiment"].positive),
        "base_negative": PhraseCounter(lexicons["base_sentiment"].negative),
        "uncertainty_positive": PhraseCounter(lexicons["uncertainty"].positive),
        "uncertainty_negative": PhraseCounter(lexicons["uncertainty"].negative),
        "gdp_growth_positive": PhraseCounter(lexicons["gdp_growth"].positive),
        "gdp_growth_negative": PhraseCounter(lexicons["gdp_growth"].negative),
        "ai_positive": PhraseCounter(lexicons["ai"].positive),
        "ai_negative": PhraseCounter(lexicons["ai"].negative),
        "unemployment_positive": PhraseCounter(lexicons["unemployment"].positive),
        "unemployment_negative": PhraseCounter(lexicons["unemployment"].negative),
        "inflation_positive": PhraseCounter(lexicons["inflation"].positive),
        "inflation_negative": PhraseCounter(lexicons["inflation"].negative),
        "rates_positive": PhraseCounter(lexicons["rates"].positive),
        "rates_negative": PhraseCounter(lexicons["rates"].negative),
        "financial_conditions_positive": PhraseCounter(lexicons["financial_conditions"].positive),
        "financial_conditions_negative": PhraseCounter(lexicons["financial_conditions"].negative),
    }


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def clean_text(text: str) -> str:
    return normalize_space(" ".join(split_sentences(text)))


def complexity_metrics(text: str) -> dict[str, float]:
    sentences = split_sentences(text)
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


def score_topic_terms(text: str, counters: dict[str, PhraseCounter]) -> dict[str, float]:
    token_count = len(tokenize(text))
    metrics = {}
    for name in ("inflation", "gdp_growth", "ai", "unemployment", "uncertainty", "rates", "financial_conditions"):
        pos = counters[f"{name}_positive"].count(text)
        neg = counters[f"{name}_negative"].count(text)
        metrics[f"{name}_net"] = safe_net(pos, neg, token_count)
    pos = counters["base_positive"].count(text)
    neg = counters["base_negative"].count(text)
    metrics["overall_net_tone"] = safe_net(pos, neg, token_count)
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
    text = clean_text(read_text(path))
    metrics = complexity_metrics(text)
    metrics.update(score_topic_terms(text, counters))
    date_text = str(meta.get("date") or "")
    if not date_text:
        match = DATE_RE.search(path.name)
        date_text = match.group(1) if match else ""
    period_year, period_quarter, year_quarter = parse_year_quarter(date_text)
    parts = path.relative_to(root).parts
    return {
        "source_file": rel_path,
        "central_bank": str(meta.get("central_bank") or (parts[0].upper() if parts else "UNKNOWN")),
        "region": str(meta.get("region") or ""),
        "doc_type": str(meta.get("doc_type") or (parts[1] if len(parts) > 1 else "")),
        "speaker": str(meta.get("speaker") or ""),
        "date": date_text,
        "period_year": period_year,
        "period_quarter": period_quarter,
        "year_quarter": year_quarter,
        **metrics,
    }


def add_category_zscores(df: pd.DataFrame, category_cols: dict[str, str]) -> pd.DataFrame:
    scored = df.copy()
    for category, source_col in category_cols.items():
        scored[f"{category}_zscore"] = safe_zscore(scored[source_col]).fillna(0.0)
    return scored


def category_zscore_long(df: pd.DataFrame, category_cols: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        for category, source_col in category_cols.items():
            rows.append({
                "period_year": row["period_year"],
                "period_quarter": row["period_quarter"],
                "year_quarter": row["year_quarter"],
                "category": category,
                "raw_score": row[source_col],
                "z_score": row[f"{category}_zscore"],
            })
    return pd.DataFrame(rows)


def plot_category_zscores(quarterly: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7.5))
    for category, col in POLICY_CATEGORY_COLS.items():
        ax.plot(
            quarterly["year_quarter"],
            quarterly[f"{category}_zscore"],
            marker="o",
            linewidth=2.1,
            markersize=4.8,
            label=category,
            color=PLOT_COLORS[category],
        )
    ax.axhline(0, color="#444444", linewidth=1, alpha=0.75)
    ax.set_title("Central Bank Policy Category Signals Over Time", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Z-Score vs. Full Policy Text Sample History", fontsize=11)
    ax.legend(ncol=3, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.tick_params(axis="x", rotation=50)
    ax.margins(x=0.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score central bank policy texts and plot category signals over time.")
    parser.add_argument("--input-root", required=True, help="Root folder containing policy text files.")
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV with central bank/date/doc-type info.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV and PNG outputs.")
    parser.add_argument("--word-bank", default="word_bank.xlsx", help="Workbook containing lexicon categories.")
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
        panel.groupby(["central_bank", "period_year", "period_quarter", "year_quarter"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["central_bank", "period_year", "period_quarter"])
    )
    quarterly = (
        panel.groupby(["period_year", "period_quarter", "year_quarter"], as_index=False)[list(POLICY_CATEGORY_COLS.values())]
        .mean()
        .sort_values(["period_year", "period_quarter"])
    )
    quarterly = add_category_zscores(quarterly, POLICY_CATEGORY_COLS)
    quarterly_long = category_zscore_long(quarterly, POLICY_CATEGORY_COLS)

    panel.to_csv(output_dir / "policy_research_doc_scores.csv", index=False)
    bank_quarter.to_csv(output_dir / "policy_research_bank_quarterly.csv", index=False)
    quarterly.to_csv(output_dir / "policy_research_category_zscores_over_time_wide.csv", index=False)
    quarterly_long.to_csv(output_dir / "policy_research_category_zscores_over_time_long.csv", index=False)
    plot_category_zscores(quarterly, output_dir / "policy_research_category_zscores_over_time.png")

    print(f"Saved {output_dir / 'policy_research_doc_scores.csv'}")
    print(f"Saved {output_dir / 'policy_research_bank_quarterly.csv'}")
    print(f"Saved {output_dir / 'policy_research_category_zscores_over_time_wide.csv'}")
    print(f"Saved {output_dir / 'policy_research_category_zscores_over_time_long.csv'}")
    print(f"Saved {output_dir / 'policy_research_category_zscores_over_time.png'}")


if __name__ == "__main__":
    main()
