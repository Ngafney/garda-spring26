import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORD_RE = re.compile(r"\b[\w%./-]+\b")

CATEGORY_LEXICONS = {
    "Inflation": {
        "positive": (
            "disinflation", "disinflationary", "inflation easing", "easing inflation",
            "lower inflation", "inflation moderated", "inflation moderation",
            "cooling inflation", "price stability", "stable prices", "lower input costs",
            "cost deflation", "commodity deflation", "freight deflation", "wage moderation",
            "lower fuel costs",
        ),
        "negative": (
            "inflation", "inflationary", "high inflation", "higher inflation",
            "rising inflation", "sticky inflation", "persistent inflation",
            "inflation pressure", "inflationary pressure", "price pressure",
            "cost inflation", "input cost inflation", "commodity inflation", "wage inflation",
            "food inflation", "fuel inflation", "rent inflation", "pass-through inflation",
            "price increases",
        ),
    },
    "GDP / Growth": {
        "positive": (
            "gdp growth", "economic growth", "real gdp", "nominal gdp",
            "gross domestic product", "economic expansion", "expanding economy",
            "growth outlook", "growth acceleration", "accelerating growth", "growth recovery",
            "soft landing", "resilient economy", "healthy economy", "consumer resilience",
            "business investment growth", "industrial production growth", "economic momentum",
            "above-trend growth", "trend growth",
        ),
        "negative": (
            "gdp decline", "negative gdp", "economic contraction", "contracting economy",
            "growth slowdown", "slowing growth", "decelerating growth", "below-trend growth",
            "weak growth", "stagnant growth", "stagnation", "recession", "recessionary",
            "hard landing", "economic weakness", "macro slowdown", "consumer slowdown",
            "industrial slowdown", "weak macro", "weaker economy",
        ),
    },
    "AI": {
        "positive": (
            "ai", "artificial intelligence", "generative ai", "gen ai", "machine learning",
            "large language model", "large language models", "llm", "llms", "ai agent",
            "ai agents", "copilot", "automation", "intelligent automation", "ai adoption",
            "ai demand", "ai infrastructure", "ai workload", "ai workloads",
            "gpu acceleration", "accelerated computing", "inference", "model training",
            "productivity gains", "efficiency gains",
        ),
        "negative": (
            "ai disruption", "ai risk", "ai risks", "ai uncertainty", "ai regulation",
            "regulatory risk", "model risk", "hallucination", "data privacy risk",
            "job displacement", "automation risk", "ai capex burden", "gpu shortage",
            "compute constraint", "ai bubble", "overinvestment in ai",
        ),
    },
    "Unemployment": {
        "positive": (
            "low unemployment", "lower unemployment", "unemployment declined",
            "unemployment fell", "job growth", "jobs growth", "payroll growth",
            "employment growth", "strong labor market", "healthy labor market",
            "labor market strength", "solid hiring", "hiring momentum", "wage growth",
            "rising employment", "workforce expansion",
        ),
        "negative": (
            "high unemployment", "higher unemployment", "unemployment rose",
            "unemployment increased", "job losses", "payroll decline", "employment decline",
            "labor market weakness", "weak labor market", "slowing hiring", "hiring slowdown",
            "layoffs", "workforce reduction", "headcount reduction", "job cuts",
            "rising unemployment", "underemployment",
        ),
    },
    "Uncertainty": {
        "positive": (
            "visibility", "clearer outlook", "predictable", "stabilizing", "well-defined",
            "certainty", "transparency", "known variables",
        ),
        "negative": (
            "uncertain", "uncertainty", "volatile", "volatility", "limited visibility",
            "challenging backdrop", "challenging environment", "macro uncertainty",
            "not clear", "unknown", "range of outcomes", "hard to predict",
            "difficult to predict", "fluid environment", "monitor closely", "cautious",
        ),
    },
}

CATEGORY_TO_COL = {
    "Inflation": "inflation_net",
    "GDP / Growth": "gdp_growth_net",
    "AI": "ai_net",
    "Unemployment": "unemployment_net",
    "Uncertainty": "uncertainty_net",
}

CALL_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def phrase_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    escaped = [re.escape(term.lower()) for term in terms]
    return re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(escaped), flags=re.IGNORECASE)


PATTERNS = {
    category: {
        sentiment: phrase_pattern(terms)
        for sentiment, terms in lexicon.items()
    }
    for category, lexicon in CATEGORY_LEXICONS.items()
}


def zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def score_file(args: tuple[str, str]) -> dict[str, object] | None:
    path_text, transcript_root_text = args
    path = Path(path_text)
    transcript_root = Path(transcript_root_text)
    rel = path.relative_to(transcript_root)
    parts = rel.parts
    if len(parts) < 4:
        return None

    year_match = re.search(r"\d{4}", parts[1])
    quarter_match = re.search(r"Q(\d)", parts[2].upper())
    if not year_match or not quarter_match:
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    token_count = len(WORD_RE.findall(text.lower()))
    if token_count == 0:
        return None

    fiscal_year = int(year_match.group(0))
    quarter_num = int(quarter_match.group(1))
    call_date_match = CALL_DATE_RE.search(path.name)
    call_date = call_date_match.group(1) if call_date_match else None
    row: dict[str, object] = {
        "company_slug": parts[0],
        "fiscal_year": fiscal_year,
        "quarter_num": quarter_num,
        "year_quarter": f"{fiscal_year}-Q{quarter_num}",
        "call_date": call_date,
        "relative_path": str(rel),
        "token_count": token_count,
    }
    for category, col in CATEGORY_TO_COL.items():
        pos = len(PATTERNS[category]["positive"].findall(text))
        neg = len(PATTERNS[category]["negative"].findall(text))
        row[col] = (pos - neg) / token_count
    return row


def plot_categories(quarterly: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7.5))
    colors = {
        "Inflation": "#C44E52",
        "GDP / Growth": "#4C72B0",
        "AI": "#55A868",
        "Unemployment": "#8172B2",
        "Uncertainty": "#CCB974",
    }
    for category, col in CATEGORY_TO_COL.items():
        ax.plot(
            quarterly["period_label"],
            quarterly[f"{col}_zscore"],
            marker="o",
            linewidth=2.1,
            markersize=4.8,
            label=category,
            color=colors[category],
        )

    ax.axhline(0, color="#444444", linewidth=1, alpha=0.75)
    ax.set_title("All Transcript Category Signals Over Time", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Z-Score vs. Full Transcript Sample History", fontsize=11)
    ax.legend(ncol=5, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.tick_params(axis="x", rotation=50)
    ax.margins(x=0.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot category lexicon z-scores over all transcripts.")
    parser.add_argument("--transcripts", default="amir/transcripts", help="Root folder containing transcript .txt files.")
    parser.add_argument("--output", default="outputs", help="Folder for CSV and PNG outputs.")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1), help="Parallel workers.")
    parser.add_argument(
        "--date-mode",
        choices=("fiscal", "call_date"),
        default="fiscal",
        help="Use fiscal folder quarter or calendar quarter derived from filename call date.",
    )
    args = parser.parse_args()

    transcript_root = Path(args.transcripts).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(transcript_root.rglob("*.txt"))
    print(f"Found {len(paths):,} transcript files under {transcript_root}")

    tasks = [(str(path), str(transcript_root)) for path in paths]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, row in enumerate(executor.map(score_file, tasks, chunksize=50), start=1):
            if row is not None:
                rows.append(row)
            if idx % 1000 == 0:
                print(f"Scored {idx:,}/{len(tasks):,} files...")

    if not rows:
        raise SystemExit("No plottable transcript files found.")

    scores = pd.DataFrame(rows)
    if args.date_mode == "call_date":
        scores["call_date_dt"] = pd.to_datetime(scores["call_date"], errors="coerce")
        scores = scores.dropna(subset=["call_date_dt"]).copy()
        scores["period_year"] = scores["call_date_dt"].dt.year
        scores["period_quarter"] = scores["call_date_dt"].dt.quarter
        scores["period_label"] = scores["period_year"].astype(str) + "-Q" + scores["period_quarter"].astype(str)
    else:
        scores["period_year"] = scores["fiscal_year"]
        scores["period_quarter"] = scores["quarter_num"]
        scores["period_label"] = scores["year_quarter"]

    quarterly = (
        scores.groupby(["period_year", "period_quarter", "period_label"], as_index=False)[list(CATEGORY_TO_COL.values())]
        .mean()
        .sort_values(["period_year", "period_quarter"])
    )

    for col in CATEGORY_TO_COL.values():
        quarterly[f"{col}_zscore"] = zscore(quarterly[col])

    long_rows = []
    for _, row in quarterly.iterrows():
        for category, col in CATEGORY_TO_COL.items():
            long_rows.append({
                "period_label": row["period_label"],
                "category": category,
                "raw_score": row[col],
                "z_score": row[f"{col}_zscore"],
            })
    long_df = pd.DataFrame(long_rows)

    prefix = "all_transcripts_call_date" if args.date_mode == "call_date" else "all_transcripts"
    scores_path = output_dir / f"{prefix}_category_scores.csv"
    wide_path = output_dir / f"{prefix}_category_zscores_over_time_wide.csv"
    long_path = output_dir / f"{prefix}_category_zscores_over_time_long.csv"
    plot_path = output_dir / f"{prefix}_category_zscores_over_time.png"

    scores.to_csv(scores_path, index=False)
    quarterly.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    plot_categories(quarterly, plot_path)

    print(f"Plottable transcripts: {len(scores):,}")
    print(f"Quarters plotted: {len(quarterly):,}")
    print(f"Plot saved to: {plot_path}")
    print(f"Long data saved to: {long_path}")
    print(f"Wide data saved to: {wide_path}")
    print(f"Transcript scores saved to: {scores_path}")


if __name__ == "__main__":
    main()
