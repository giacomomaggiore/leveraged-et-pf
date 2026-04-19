from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from orchestration import (
    CompleteSimulationResult,
    MarketDataConfig,
    MonteCarloConfig,
    PortfolioConfig,
    SimulationConfig,
    SpotAssetConfig,
    SyntheticLETFAssetConfig,
    _canonical_portfolio_name,
    run_complete_simulation,
    save_portfolio_metrics_summary,
)
from visuals import plot_drawdown_chart, plot_spaghetti_paths, plot_terminal_wealth_distribution


DEFAULT_TER = 0.0092
DEFAULT_SPREAD = 0.0030
OUTPUT_DIR = Path("output_2")
AGGREGATE_CSV = OUTPUT_DIR / "portfolio_metrics_summary.csv"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "portfolio"


def _build_assets_and_weights(definitions: list[dict]) -> tuple[list[SpotAssetConfig | SyntheticLETFAssetConfig], dict[str, float]]:
    assets: list[SpotAssetConfig | SyntheticLETFAssetConfig] = []
    target_weights: dict[str, float] = {}

    for entry in definitions:
        asset_id = str(entry["id"])
        ticker = str(entry["ticker"])
        weight = float(entry["weight"])
        leverage = float(entry.get("leverage", 1.0))
        ter = float(entry.get("ter", DEFAULT_TER))
        spread = float(entry.get("spread", DEFAULT_SPREAD))

        if leverage == 1.0:
            assets.append(SpotAssetConfig(id=asset_id, ticker=ticker))
        else:
            assets.append(
                SyntheticLETFAssetConfig(
                    id=asset_id,
                    underlying_ticker=ticker,
                    leverage=leverage,
                    ter=ter,
                    spread=spread,
                )
            )

        target_weights[asset_id] = weight

    total_weight = sum(target_weights.values())
    if abs(total_weight - 1.0) > 1e-10:
        raise ValueError(f"Target weights must sum to 1.0, got {total_weight:.6f}")

    if len(target_weights) != len(definitions):
        raise ValueError("Each portfolio asset id must be unique.")

    return assets, target_weights


def _base_config(assets: list[SpotAssetConfig | SyntheticLETFAssetConfig], target_weights: dict[str, float]) -> SimulationConfig:
    return SimulationConfig(
        market=MarketDataConfig(
            start="2018-04-03",
            end="2025-12-31",
            fred_series="SOFR",
            fred_is_percent=True,
        ),
        assets=assets,
        portfolio=PortfolioConfig(
            target_weights=target_weights,
            initial_capital=100_000.0,
            rebalance_frequency_days=252,
            tolerance_band=0.05,
            capital_gains_tax_rate=0.26,
        ),
        monte_carlo=MonteCarloConfig(
            n_paths=10000,
            horizon_days=2520,
            method="bootstrap",
            distribution="normal",
            student_t_df=6.0,
            seed=42,
        ),
        metrics_ruin_threshold_fraction=0.10,
        use_mean_risk_free_for_metrics=True,
    )


def _set_portfolio_name(csv_path: Path, config: SimulationConfig, portfolio_name: str) -> None:
    composition = _canonical_portfolio_name(config)
    df = pd.read_csv(csv_path)

    if "portfolio_name" not in df.columns:
        df.insert(1, "portfolio_name", "")
    # Keep labels as string dtype to avoid pandas incompatible assignment warnings.
    df["portfolio_name"] = df["portfolio_name"].fillna("").astype(str)

    match = df["portfolio composition"] == composition
    if not bool(match.any()):
        raise ValueError(f"Could not find composition row in {csv_path}")

    df.loc[match, "portfolio_name"] = portfolio_name
    df.to_csv(csv_path, index=False)


def _save_plotly_figure(fig, png_path: Path, html_path: Path) -> None:
    # Always persist an interactive HTML, then try static PNG export.
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path, format="png", scale=2)
    except Exception as exc:
        print(f"    Warning: could not export PNG '{png_path.name}' ({exc}). HTML was saved.")


def _build_assets_subtitle(config: SimulationConfig) -> str:
    rows: list[str] = []
    for asset in config.assets:
        weight = float(config.portfolio.target_weights.get(asset.id, 0.0))
        if isinstance(asset, SpotAssetConfig):
            rows.append(f"{asset.ticker} | weight={weight:.1%}")
        else:
            rows.append(
                f"{asset.underlying_ticker} {asset.leverage:.1f}x | weight={weight:.1%} | "
                f"TER={asset.ter:.2%} | spread={asset.spread:.2%}"
            )
    return "<br>".join(rows)


def _export_figures(
    *,
    portfolio_name: str,
    config: SimulationConfig,
    result: CompleteSimulationResult,
    output_dir: Path,
) -> None:
    wealth_paths = result.portfolio.wealth_paths
    terminal = wealth_paths[:, -1]
    subtitle = _build_assets_subtitle(config)

    summary_note = (
        f"FRED={config.market.fred_series}"
        "<br>"
        f"{config.monte_carlo.n_paths:,} draws | {config.monte_carlo.horizon_days} horizon days"
        "<br>"
        f"Method={config.monte_carlo.method} | Distribution={config.monte_carlo.distribution}"
        "<br>"
        f"Initial capital={config.portfolio.initial_capital:,.2f} | Rebalance={config.portfolio.rebalance_frequency_days} days"
        "<br>"
        f"Tolerance={config.portfolio.tolerance_band:.2%} | Tax={config.portfolio.capital_gains_tax_rate:.2%}"
    )

    fig_spaghetti = plot_spaghetti_paths(
        wealth_paths=wealth_paths,
        n_sample=100,
        seed=42,
        normalize_to_1=False,
        title=f"Monte Carlo Spaghetti - {portfolio_name}",
        subtitle=subtitle,
        bottom_note=summary_note,
        subtitle_align="left",
        bottom_note_align="left",
        bottom_note_x=0.0,
        bottom_note_y=-0.15,
        bottom_note_box=True,
        backend="plotly",
        width=1200,
        height=800,
    )

    fig_terminal = plot_terminal_wealth_distribution(
        wealth_paths=wealth_paths,
        bins=60,
        title=f"Terminal Wealth Distribution - {portfolio_name}",
        backend="plotly",
        width=1200,
        height=800,
    )

    fig_drawdown = plot_drawdown_chart(
        wealth_paths=wealth_paths,
        drawdowns=result.metrics.drawdowns,
        title=f"Drawdown Chart - {portfolio_name}",
        backend="plotly",
        width=1200,
        height=800,
    )

    terminal_note = (
        f"min={float(np.min(terminal)):,.0f} | p5={float(np.quantile(terminal, 0.05)):,.0f} | "
        f"median={float(np.median(terminal)):,.0f} | mean={float(np.mean(terminal)):,.0f} | "
        f"p95={float(np.quantile(terminal, 0.95)):,.0f} | max={float(np.max(terminal)):,.0f}"
    )
    fig_terminal.add_annotation(
        x=0.0,
        y=-0.18,
        xref="paper",
        yref="paper",
        text=terminal_note,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
    )

    _save_plotly_figure(
        fig_spaghetti,
        output_dir / "spaghetti.png",
        output_dir / "spaghetti.html",
    )
    _save_plotly_figure(
        fig_terminal,
        output_dir / "terminal_distribution.png",
        output_dir / "terminal_distribution.html",
    )
    _save_plotly_figure(
        fig_drawdown,
        output_dir / "drawdown.png",
        output_dir / "drawdown.html",
    )


PORTFOLIOS = [
    {
        "name": "EQUITY_WORLD_100_VWCE",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 1.0, "leverage": 1.0},
        ],
    },
    {
        "name": "EQUITY_WORLD_100_VWCE_x1_5",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 1.0, "leverage": 1.5},
        ],
    },
    {
        "name": "EQUITY_WORLD_100_VWCE_x2",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 1.0, "leverage": 2.0},
        ],
    },
    {
        "name": "EQUITY_WORLD_100_VWCE_x3",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 1.0, "leverage": 3.0},
        ],
    },
    {
        "name": "CLASSIC_60_40",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.60, "leverage": 1.0},
            {"id": "IGLA", "ticker": "IGLA.L", "weight": 0.40, "leverage": 1.0},
        ],
    },
    {
        "name": "CLASSIC_60_40_x2_x2",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.60, "leverage": 2.0},
            {"id": "IGLA", "ticker": "IGLA.L", "weight": 0.40, "leverage": 2.0},
        ],
    },
    {
        "name": "CLASSIC_60_40_VWCE_x2_IGLA_spot",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.60, "leverage": 2.0},
            {"id": "IGLA", "ticker": "IGLA.L", "weight": 0.40, "leverage": 1.0},
        ],
    },
    {
        "name": "ALL_WEATHER",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.30, "leverage": 1.0},
            {"id": "UTHY", "ticker": "UTHY", "weight": 0.40, "leverage": 1.0},
            {"id": "LTPZ", "ticker": "LTPZ", "weight": 0.15, "leverage": 1.0},
            {"id": "GC", "ticker": "GC=F", "weight": 0.075, "leverage": 1.0},
            {"id": "PDBC", "ticker": "PDBC", "weight": 0.075, "leverage": 1.0},
        ],
    },
    {
        "name": "ALL_WEATHER_x2",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.30, "leverage": 2.0},
            {"id": "UTHY", "ticker": "UTHY", "weight": 0.40, "leverage": 2.0},
            {"id": "LTPZ", "ticker": "LTPZ", "weight": 0.15, "leverage": 2.0},
            {"id": "GC", "ticker": "GC=F", "weight": 0.075, "leverage": 2.0},
            {"id": "PDBC", "ticker": "PDBC", "weight": 0.075, "leverage": 2.0},
        ],
    },
    {
        "name": "ALL_WEATHER_x3",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.30, "leverage": 3.0},
            {"id": "UTHY", "ticker": "UTHY", "weight": 0.40, "leverage": 3.0},
            {"id": "LTPZ", "ticker": "LTPZ", "weight": 0.15, "leverage": 3.0},
            {"id": "GC", "ticker": "GC=F", "weight": 0.075, "leverage": 3.0},
            {"id": "PDBC", "ticker": "PDBC", "weight": 0.075, "leverage": 3.0},
        ],
    },
    {
        "name": "GOLDEN_BUTTERFLY",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.20, "leverage": 1.0},
            {"id": "EMUL", "ticker": "EMUL.MI", "weight": 0.20, "leverage": 1.0},
            {"id": "UTHY", "ticker": "UTHY", "weight": 0.20, "leverage": 1.0},
            {"id": "GOVT", "ticker": "GOVT", "weight": 0.20, "leverage": 1.0},
            {"id": "GC", "ticker": "GC=F", "weight": 0.20, "leverage": 1.0},
        ],
    },
    {
        "name": "GOLDEN_BUTTERFLY_x2_mixed",
        "assets": [
            {"id": "VWCE", "ticker": "VWCE.MI", "weight": 0.20, "leverage": 2.0},
            {"id": "EMUL", "ticker": "EMUL.MI", "weight": 0.20, "leverage": 2.0},
            {"id": "UTHY", "ticker": "UTHY", "weight": 0.20, "leverage": 2.0},
            {"id": "GOVT", "ticker": "GOVT", "weight": 0.20, "leverage": 1.0},
            {"id": "GC", "ticker": "GC=F", "weight": 0.20, "leverage": 1.0},
        ],
    },
]


def run_batch() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, portfolio in enumerate(PORTFOLIOS, start=1):
        name = str(portfolio["name"])
        definitions = portfolio["assets"]

        print(f"[{idx}/{len(PORTFOLIOS)}] Running: {name}")

        assets, target_weights = _build_assets_and_weights(definitions)
        config = _base_config(assets, target_weights)
        result = run_complete_simulation(config)

        # 1) Upsert row in global /output_2/portfolio_metrics_summary.csv
        save_portfolio_metrics_summary(
            config=config,
            metrics_summary=result.metrics.summary,
            output_csv_path=AGGREGATE_CSV,
        )
        _set_portfolio_name(AGGREGATE_CSV, config, name)

        # 2) Save one-row CSV per portfolio under /output_2/<portfolio>/portfolio_metrics_summary.csv
        per_portfolio_dir = OUTPUT_DIR / _slugify(name)
        per_portfolio_dir.mkdir(parents=True, exist_ok=True)
        per_portfolio_csv = per_portfolio_dir / "portfolio_metrics_summary.csv"
        save_portfolio_metrics_summary(
            config=config,
            metrics_summary=result.metrics.summary,
            output_csv_path=per_portfolio_csv,
        )
        _set_portfolio_name(per_portfolio_csv, config, name)

        # Save the raw metrics table too for easier inspection.
        result.metrics.summary.to_csv(per_portfolio_dir / "metrics_summary_table.csv")

        # 3) Save figures in each portfolio output folder.
        _export_figures(
            portfolio_name=name,
            config=config,
            result=result,
            output_dir=per_portfolio_dir,
        )

        print(f"    Saved: {AGGREGATE_CSV}")
        print(f"    Saved: {per_portfolio_csv}")
        print(f"    Saved figures under: {per_portfolio_dir}")


if __name__ == "__main__":
    run_batch()