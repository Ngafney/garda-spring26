from __future__ import annotations

"""
Saved macro-analysis pack for the earnings-call sentiment panel.

Outputs:
- merged monthly/quarterly analysis datasets
- regression tables
- a bundle of saved PNG charts for macro and trading research
- a short markdown summary of the strongest relationships found
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
SENTIMENT_DIR = ROOT_DIR / "outputs" / "research_sentiment"
MACRO_WIDE_PATH = ROOT_DIR.parent / "data" / "macro" / "fred_macro_wide.csv"
POLICY_SENTENCES_PATH = ROOT_DIR.parent / "data" / "policy" / "outputs" / "scored_sentences.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "research_sentiment_analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

MONTHLY_SIGNAL_COLUMNS = [
    "ew_growth_signal",
    "ew_margin_signal",
    "ew_composite_signal",
    "ew_macro_risk_density",
    "ew_pricing_power_net",
    "ew_labor_pressure_density",
    "ew_supply_chain_pressure_density",
    "ew_uncertainty_density",
    "share_guidance_raised",
    "share_guidance_lowered",
    "share_positive_composite",
]
REGRESSION_FEATURES = [
    "ew_growth_signal",
    "ew_margin_signal",
    "ew_pricing_power_net",
    "ew_labor_pressure_density",
    "ew_supply_chain_pressure_density",
    "ew_macro_risk_density",
    "ew_uncertainty_density",
    "share_guidance_raised",
    "fed_inflation_score",
    "fed_growth_score",
]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def period_to_timestamp(series: pd.Series, freq: str) -> pd.Series:
    return pd.PeriodIndex(series.astype(str), freq=freq).to_timestamp()


def load_sentiment_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(SENTIMENT_DIR / "earnings_research_macro_monthly.csv")
    quarterly = pd.read_csv(SENTIMENT_DIR / "earnings_research_macro_quarterly.csv")
    latest = pd.read_csv(SENTIMENT_DIR / "earnings_research_latest_snapshot.csv")

    monthly["month"] = period_to_timestamp(monthly["call_month"], "M")
    quarterly["quarter"] = pd.PeriodIndex(quarterly["call_quarter"].astype(str), freq="Q")
    return monthly, quarterly, latest


def load_fred_macro() -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = pd.read_csv(MACRO_WIDE_PATH)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").reset_index(drop=True)

    macro["cpi_yoy"] = macro["CPI"].pct_change(12, fill_method=None) * 100
    macro["core_cpi_yoy"] = macro["Core_CPI"].pct_change(12, fill_method=None) * 100
    macro["cpi_mom_annualized_3m"] = ((macro["CPI"] / macro["CPI"].shift(3)) ** 4 - 1) * 100
    macro["core_cpi_mom_annualized_3m"] = ((macro["Core_CPI"] / macro["Core_CPI"].shift(3)) ** 4 - 1) * 100
    macro["unemployment_change_3m"] = macro["Unemployment_Rate"] - macro["Unemployment_Rate"].shift(3)
    macro["unemployment_change_12m"] = macro["Unemployment_Rate"] - macro["Unemployment_Rate"].shift(12)

    monthly_macro = macro.rename(columns={"date": "month"})

    quarterly_macro = macro.loc[macro["GDP_YoY_Pct"].notna(), ["date", "GDP_YoY_Pct"]].copy()
    quarterly_macro["quarter"] = quarterly_macro["date"].dt.to_period("Q")
    quarterly_macro = quarterly_macro[["quarter", "GDP_YoY_Pct"]].drop_duplicates("quarter")
    return monthly_macro, quarterly_macro


def load_policy_monthly() -> pd.DataFrame:
    sentences = pd.read_csv(POLICY_SENTENCES_PATH)
    sentences["date"] = pd.to_datetime(sentences["date"])
    sentences["month"] = sentences["date"].dt.to_period("M").dt.to_timestamp()
    fed = sentences.loc[sentences["central_bank"].eq("Fed")].copy()

    if fed.empty:
        return pd.DataFrame(columns=["month"])

    grouped = fed.groupby("month").agg(
        fed_all_score=("weighted_score", "mean"),
        fed_sentence_count=("weighted_score", "size"),
        fed_document_count=("doc_id", "nunique"),
    )

    categories = {"inflation": "fed_inflation_score", "growth": "fed_growth_score", "labor": "fed_labor_score", "guidance": "fed_guidance_score"}
    for keyword, column in categories.items():
        mask = fed["categories"].fillna("").str.contains(rf"\b{keyword}\b", case=False, regex=True)
        category_frame = fed.loc[mask].groupby("month")["weighted_score"].mean().rename(column)
        grouped = grouped.join(category_frame, how="left")

    return grouped.reset_index().sort_values("month")


def build_monthly_analysis(monthly: pd.DataFrame, macro: pd.DataFrame, fed: pd.DataFrame) -> pd.DataFrame:
    merged = monthly.merge(macro, on="month", how="left").merge(fed, on="month", how="left")
    merged = merged.sort_values("month").reset_index(drop=True)

    merged["guidance_balance"] = merged["share_guidance_raised"] - merged["share_guidance_lowered"]
    merged["cpi_yoy_3m_fwd"] = merged["cpi_yoy"].shift(-3)
    merged["core_cpi_yoy_3m_fwd"] = merged["core_cpi_yoy"].shift(-3)
    merged["cpi_yoy_change_3m_fwd"] = merged["cpi_yoy"].shift(-3) - merged["cpi_yoy"]
    merged["core_cpi_yoy_change_3m_fwd"] = merged["core_cpi_yoy"].shift(-3) - merged["core_cpi_yoy"]
    merged["unemployment_rate_3m_fwd"] = merged["Unemployment_Rate"].shift(-3)
    merged["unemployment_change_3m_fwd"] = merged["Unemployment_Rate"].shift(-3) - merged["Unemployment_Rate"]

    for column in MONTHLY_SIGNAL_COLUMNS + ["cpi_yoy", "core_cpi_yoy", "Unemployment_Rate", "fed_inflation_score", "fed_growth_score"]:
        if column in merged.columns:
            merged[f"{column}_z"] = zscore(merged[column])
    return merged


def build_quarterly_analysis(quarterly: pd.DataFrame, quarterly_macro: pd.DataFrame) -> pd.DataFrame:
    merged = quarterly.merge(quarterly_macro, on="quarter", how="left")
    merged = merged.sort_values("quarter").reset_index(drop=True)
    merged["gdp_yoy_next_q"] = merged["GDP_YoY_Pct"].shift(-1)
    merged["gdp_yoy_change_next_q"] = merged["GDP_YoY_Pct"].shift(-1) - merged["GDP_YoY_Pct"]
    return merged


def fit_standardized_ols(frame: pd.DataFrame, target: str, features: list[str]) -> tuple[sm.regression.linear_model.RegressionResultsWrapper | None, pd.DataFrame]:
    cols = [target, *features]
    sample = frame[cols].dropna().copy()
    if len(sample) < max(12, len(features) + 3):
        return None, pd.DataFrame()

    y = zscore(sample[target])
    X = sample[features].apply(zscore)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    rows = []
    for feature in ["const", *features]:
        rows.append(
            {
                "target": target,
                "feature": feature,
                "coef": float(model.params.get(feature, np.nan)),
                "t_stat": float(model.tvalues.get(feature, np.nan)),
                "p_value": float(model.pvalues.get(feature, np.nan)),
                "r_squared": float(model.rsquared),
                "n_obs": int(model.nobs),
            }
        )
    return model, pd.DataFrame(rows)


def run_univariate_regressions(frame: pd.DataFrame, targets: list[str], features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for target in targets:
        for feature in features:
            sample = frame[[target, feature]].dropna().copy()
            if len(sample) < 12:
                continue
            y = zscore(sample[target])
            X = sm.add_constant(zscore(sample[feature]).rename(feature))
            model = sm.OLS(y, X).fit(cov_type="HC1")
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "coef": float(model.params[feature]),
                    "t_stat": float(model.tvalues[feature]),
                    "p_value": float(model.pvalues[feature]),
                    "r_squared": float(model.rsquared),
                    "n_obs": int(model.nobs),
                }
            )
    return pd.DataFrame(rows)


def lead_lag_correlation(series_x: pd.Series, series_y: pd.Series, max_lead: int = 6) -> dict[int, float]:
    values: dict[int, float] = {}
    for lead in range(-max_lead, max_lead + 1):
        aligned = pd.concat([series_x, series_y.shift(-lead)], axis=1).dropna()
        values[lead] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) >= 6 else np.nan
    return values


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_signal_breadth(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(monthly["month"], monthly["ew_composite_signal"], label="EW Composite", color="#0b6e4f")
    axes[0].plot(monthly["month"], monthly["tw_composite_signal"], label="Token-Weighted Composite", color="#c84c09", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Corporate Earnings Sentiment Composite")
    axes[0].legend()

    axes[1].plot(monthly["month"], monthly["share_positive_composite"], color="#1d4e89", label="Share Positive Composite")
    axes[1].plot(monthly["month"], monthly["share_guidance_raised"], color="#2a9d8f", label="Share Guidance Raised")
    axes[1].plot(monthly["month"], monthly["share_guidance_lowered"], color="#d62828", label="Share Guidance Lowered")
    axes[1].set_title("Breadth and Guidance")
    axes[1].legend()

    axes[2].plot(monthly["month"], monthly["ew_growth_signal"], color="#264653", label="Growth")
    axes[2].plot(monthly["month"], monthly["ew_margin_signal"], color="#e9c46a", label="Margin")
    axes[2].plot(monthly["month"], monthly["ew_macro_risk_density"], color="#b56576", label="Macro Risk")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Signal Mix")
    axes[2].legend()
    savefig(PLOTS_DIR / "01_signal_breadth_and_mix.png")


def plot_inflation_vs_pricing(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(monthly["month"], monthly["cpi_yoy"], label="CPI YoY", color="#d62828")
    axes[0].plot(monthly["month"], monthly["core_cpi_yoy"], label="Core CPI YoY", color="#7f5539")
    ax2 = axes[0].twinx()
    ax2.plot(monthly["month"], monthly["ew_pricing_power_net"], label="Pricing Power Net", color="#1d3557")
    ax2.plot(monthly["month"], monthly["ew_supply_chain_pressure_density"], label="Supply Pressure", color="#457b9d", alpha=0.8)
    axes[0].set_title("Inflation vs Corporate Pricing and Supply Signals")
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1].plot(monthly["month"], monthly["fed_inflation_score"], label="Fed Inflation Score", color="#6a4c93")
    axes[1].plot(monthly["month"], monthly["ew_pricing_power_net"], label="Corporate Pricing Power", color="#1d3557")
    axes[1].plot(monthly["month"], monthly["ew_macro_risk_density"], label="Corporate Macro Risk", color="#b56576")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Fed Inflation Language vs Corporate Inflation Pass-Through")
    axes[1].legend()
    savefig(PLOTS_DIR / "02_inflation_vs_pricing.png")


def plot_growth_and_labor(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    axes[0].plot(monthly["month"], monthly["ew_growth_signal"], label="Corporate Growth Signal", color="#0b6e4f")
    axes[0].plot(monthly["month"], monthly["ew_uncertainty_density"], label="Corporate Uncertainty", color="#8d99ae")
    ax2 = axes[0].twinx()
    ax2.plot(monthly["month"], monthly["Unemployment_Rate"], label="Unemployment Rate", color="#d62828")
    axes[0].set_title("Corporate Growth and Uncertainty vs Unemployment")
    axes[0].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1].plot(monthly["month"], monthly["ew_labor_pressure_density"], label="Labor Pressure", color="#b56576")
    axes[1].plot(monthly["month"], monthly["ew_margin_signal"], label="Margin Signal", color="#e9c46a")
    ax3 = axes[1].twinx()
    ax3.plot(monthly["month"], monthly["unemployment_change_3m"], label="3m Unemployment Change", color="#264653")
    axes[1].set_title("Labor Pressure vs Labor-Market Slack")
    axes[1].legend(loc="upper left")
    ax3.legend(loc="upper right")

    qx = quarterly["quarter"].astype(str)
    axes[2].plot(qx, quarterly["ew_growth_signal"], marker="o", label="Quarterly Growth Signal", color="#0b6e4f")
    axes[2].plot(qx, quarterly["GDP_YoY_Pct"], marker="o", label="GDP YoY", color="#1d3557")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title("Quarterly Corporate Growth Signal vs GDP YoY")
    axes[2].legend()
    savefig(PLOTS_DIR / "03_growth_and_labor.png")


def plot_macro_dashboard(monthly: pd.DataFrame) -> None:
    dashboard = monthly[[
        "month",
        "ew_growth_signal_z",
        "ew_margin_signal_z",
        "ew_macro_risk_density_z",
        "ew_pricing_power_net_z",
        "cpi_yoy_z",
        "core_cpi_yoy_z",
        "Unemployment_Rate_z",
        "fed_inflation_score_z",
    ]].copy()
    renamed = dashboard.rename(
        columns={
            "ew_growth_signal_z": "Corp Growth",
            "ew_margin_signal_z": "Corp Margin",
            "ew_macro_risk_density_z": "Corp Macro Risk",
            "ew_pricing_power_net_z": "Corp Pricing",
            "cpi_yoy_z": "CPI YoY",
            "core_cpi_yoy_z": "Core CPI YoY",
            "Unemployment_Rate_z": "Unemployment",
            "fed_inflation_score_z": "Fed Inflation Text",
        }
    )
    fig, ax = plt.subplots(figsize=(15, 7))
    for column in renamed.columns[1:]:
        ax.plot(renamed["month"], renamed[column], label=column)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Standardized Macro Dashboard")
    ax.legend(ncol=4, fontsize=9)
    savefig(PLOTS_DIR / "04_standardized_macro_dashboard.png")


def plot_scatter_with_fit(frame: pd.DataFrame, x: str, y: str, title: str, path: str) -> None:
    sample = frame[[x, y]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=sample, x=x, y=y, scatter_kws={"alpha": 0.75, "s": 50}, line_kws={"color": "#d62828"}, ax=ax)
    corr = sample[x].corr(sample[y]) if len(sample) >= 3 else np.nan
    ax.set_title(f"{title}\nCorrelation = {corr:.2f}, N = {len(sample)}")
    savefig(PLOTS_DIR / path)


def plot_correlation_heatmap(monthly: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ew_growth_signal",
        "ew_margin_signal",
        "ew_pricing_power_net",
        "ew_labor_pressure_density",
        "ew_supply_chain_pressure_density",
        "ew_macro_risk_density",
        "ew_uncertainty_density",
        "cpi_yoy",
        "core_cpi_yoy",
        "Unemployment_Rate",
        "fed_inflation_score",
        "fed_growth_score",
    ]
    corr = monthly[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Contemporaneous Correlation Heatmap")
    savefig(PLOTS_DIR / "08_contemporaneous_correlation_heatmap.png")
    return corr


def plot_lead_lag_heatmap(monthly: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "Pricing -> CPI YoY": ("ew_pricing_power_net", "cpi_yoy"),
        "Pricing -> Core CPI YoY": ("ew_pricing_power_net", "core_cpi_yoy"),
        "Labor Pressure -> Unemployment": ("ew_labor_pressure_density", "Unemployment_Rate"),
        "Growth -> Unemployment": ("ew_growth_signal", "Unemployment_Rate"),
        "Macro Risk -> CPI YoY": ("ew_macro_risk_density", "cpi_yoy"),
        "Fed Inflation -> CPI YoY": ("fed_inflation_score", "cpi_yoy"),
    }
    rows = {}
    for label, (x, y) in pairs.items():
        rows[label] = lead_lag_correlation(monthly[x], monthly[y], max_lead=6)

    lead_lag = pd.DataFrame(rows).T
    lead_lag.columns = [f"lead_{col:+d}m" for col in lead_lag.columns]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(lead_lag, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Lead/Lag Correlation Heatmap\nPositive leads mean transcript/policy signal leads macro")
    savefig(PLOTS_DIR / "09_lead_lag_heatmap.png")
    return lead_lag


def plot_regression_betas(regression_table: pd.DataFrame) -> None:
    keep_targets = [
        "cpi_yoy_change_3m_fwd",
        "core_cpi_yoy_change_3m_fwd",
        "unemployment_change_3m_fwd",
    ]
    subset = regression_table.loc[
        regression_table["target"].isin(keep_targets) & regression_table["feature"].ne("const")
    ].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, target in zip(axes, keep_targets):
        frame = subset.loc[subset["target"].eq(target)].sort_values("coef")
        ax.barh(frame["feature"], frame["coef"], color=np.where(frame["coef"] >= 0, "#2a9d8f", "#d62828"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(target)
    savefig(PLOTS_DIR / "10_regression_betas_monthly.png")


def plot_inflation_regimes(monthly: pd.DataFrame) -> None:
    sample = monthly[[
        "cpi_yoy",
        "ew_pricing_power_net",
        "ew_macro_risk_density",
        "ew_margin_signal",
        "ew_growth_signal",
    ]].dropna().copy()
    sample["inflation_regime"] = pd.qcut(sample["cpi_yoy"], q=3, labels=["Low CPI", "Mid CPI", "High CPI"])
    melted = sample.melt(id_vars="inflation_regime", var_name="signal", value_name="value")

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(data=melted, x="signal", y="value", hue="inflation_regime", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Corporate Signal Distributions by Inflation Regime")
    savefig(PLOTS_DIR / "11_inflation_regime_boxplots.png")


def write_summary(
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    univariate: pd.DataFrame,
    multivariate: pd.DataFrame,
    lead_lag: pd.DataFrame,
) -> None:
    strongest_uni = univariate.sort_values("r_squared", ascending=False).head(10)
    strongest_lead = lead_lag.stack().sort_values(key=lambda s: s.abs(), ascending=False).head(10)

    lines = [
        "# Research Sentiment Macro Analysis",
        "",
        "## Coverage",
        f"- Monthly observations: {len(monthly)}",
        f"- Quarterly observations: {len(quarterly)}",
        f"- Month range: {monthly['month'].min().date()} to {monthly['month'].max().date()}",
        "",
        "## Recent Macro Signal Read",
        f"- Latest monthly composite: {monthly.iloc[-1]['ew_composite_signal']:.4f}",
        f"- Latest monthly growth signal: {monthly.iloc[-1]['ew_growth_signal']:.4f}",
        f"- Latest monthly pricing-power signal: {monthly.iloc[-1]['ew_pricing_power_net']:.4f}",
        f"- Latest monthly macro-risk density: {monthly.iloc[-1]['ew_macro_risk_density']:.4f}",
        "",
        "## Strongest Univariate Fits By R-squared",
    ]
    for row in strongest_uni.itertuples(index=False):
        lines.append(
            f"- {row.feature} -> {row.target}: coef={row.coef:.3f}, t={row.t_stat:.2f}, "
            f"p={row.p_value:.3f}, R^2={row.r_squared:.3f}, n={row.n_obs}"
        )

    lines.extend(["", "## Strongest Lead/Lag Correlations"])
    for (pair, lead), value in strongest_lead.items():
        lines.append(f"- {pair} at {lead}: corr={value:.3f}")

    lines.extend(["", "## Multivariate Monthly Models"])
    for target in multivariate["target"].drop_duplicates():
        subset = multivariate.loc[multivariate["target"].eq(target) & multivariate["feature"].ne("const")]
        top = subset.reindex(subset["coef"].abs().sort_values(ascending=False).index).head(5)
        lines.append(f"- {target}:")
        for row in top.itertuples(index=False):
            lines.append(f"  {row.feature}: coef={row.coef:.3f}, t={row.t_stat:.2f}, p={row.p_value:.3f}")

    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    monthly_sentiment, quarterly_sentiment, latest = load_sentiment_data()
    monthly_macro, quarterly_macro = load_fred_macro()
    fed_policy = load_policy_monthly()

    monthly = build_monthly_analysis(monthly_sentiment, monthly_macro, fed_policy)
    quarterly = build_quarterly_analysis(quarterly_sentiment, quarterly_macro)

    monthly.to_csv(TABLES_DIR / "monthly_macro_analysis_dataset.csv", index=False)
    quarterly.to_csv(TABLES_DIR / "quarterly_macro_analysis_dataset.csv", index=False)
    latest.to_csv(TABLES_DIR / "latest_snapshot_copy.csv", index=False)

    plot_signal_breadth(monthly)
    plot_inflation_vs_pricing(monthly)
    plot_growth_and_labor(monthly, quarterly)
    plot_macro_dashboard(monthly)
    plot_scatter_with_fit(monthly, "ew_pricing_power_net", "cpi_yoy_3m_fwd", "Pricing Power vs 3-Month Forward CPI YoY", "05_scatter_pricing_vs_future_cpi.png")
    plot_scatter_with_fit(monthly, "ew_pricing_power_net", "core_cpi_yoy_3m_fwd", "Pricing Power vs 3-Month Forward Core CPI YoY", "06_scatter_pricing_vs_future_core_cpi.png")
    plot_scatter_with_fit(monthly, "ew_labor_pressure_density", "unemployment_change_3m_fwd", "Labor Pressure vs 3-Month Forward Unemployment Change", "07_scatter_labor_vs_future_unemployment.png")
    plot_scatter_with_fit(quarterly, "ew_growth_signal", "gdp_yoy_next_q", "Quarterly Growth Signal vs Next-Quarter GDP YoY", "12_scatter_growth_vs_next_gdp.png")
    corr = plot_correlation_heatmap(monthly)
    lead_lag = plot_lead_lag_heatmap(monthly)

    univariate_monthly = run_univariate_regressions(
        monthly,
        targets=[
            "cpi_yoy_3m_fwd",
            "core_cpi_yoy_3m_fwd",
            "cpi_yoy_change_3m_fwd",
            "core_cpi_yoy_change_3m_fwd",
            "unemployment_change_3m_fwd",
        ],
        features=REGRESSION_FEATURES,
    )
    univariate_quarterly = run_univariate_regressions(
        quarterly,
        targets=["gdp_yoy_next_q", "gdp_yoy_change_next_q"],
        features=["ew_growth_signal", "ew_margin_signal", "ew_composite_signal", "share_guidance_raised"],
    )
    univariate_monthly.to_csv(TABLES_DIR / "monthly_univariate_regressions.csv", index=False)
    univariate_quarterly.to_csv(TABLES_DIR / "quarterly_univariate_regressions.csv", index=False)
    corr.to_csv(TABLES_DIR / "contemporaneous_correlation_matrix.csv")
    lead_lag.to_csv(TABLES_DIR / "lead_lag_correlations.csv")

    multivariate_frames: list[pd.DataFrame] = []
    for target in ["cpi_yoy_change_3m_fwd", "core_cpi_yoy_change_3m_fwd", "unemployment_change_3m_fwd"]:
        _, result = fit_standardized_ols(monthly, target, REGRESSION_FEATURES)
        if not result.empty:
            multivariate_frames.append(result)
    multivariate = pd.concat(multivariate_frames, ignore_index=True) if multivariate_frames else pd.DataFrame()
    multivariate.to_csv(TABLES_DIR / "monthly_multivariate_regressions.csv", index=False)

    plot_regression_betas(multivariate)
    plot_inflation_regimes(monthly)
    write_summary(monthly, quarterly, univariate_monthly, multivariate, lead_lag)

    LOGGER.info("Saved plots to %s", PLOTS_DIR)
    LOGGER.info("Saved tables to %s", TABLES_DIR)
    LOGGER.info("Wrote markdown summary to %s", OUTPUT_DIR / "analysis_summary.md")


if __name__ == "__main__":
    main()
