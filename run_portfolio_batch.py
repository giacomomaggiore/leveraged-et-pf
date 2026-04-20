from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from visuals import plot_spaghetti_paths, plot_terminal_wealth_distribution


DEFAULT_TER = 0.0092
DEFAULT_SPREAD = 0.0030
OUTPUT_DIR = Path("output")
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


def _cleanup_legacy_exports(output_dir: Path) -> None:
    legacy_files = [
        "spaghetti.png",
        "spaghetti.html",
        "terminal_distribution.png",
        "terminal_distribution.html",
        "drawdown.png",
        "drawdown.html",
        "metrics_summary_table.csv",
    ]
    for filename in legacy_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()


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

    terminal_summary = pd.Series(
        {
            "min": float(np.min(terminal)),
            "p5": float(np.quantile(terminal, 0.05)),
            "median": float(np.median(terminal)),
            "mean": float(np.mean(terminal)),
            "p95": float(np.quantile(terminal, 0.95)),
            "max": float(np.max(terminal)),
        },
        name="TerminalWealth",
    )

    summary_note = (
        f"FRED={config.market.fred_series}"
        "<br>"
        f"{config.monte_carlo.n_paths:,} draws | {config.monte_carlo.horizon_days} horizon days"
        "<br>"
        f"{config.monte_carlo.method} Method<br>Distribution={config.monte_carlo.distribution}<br>Student_t_df={config.monte_carlo.student_t_df}"
        "<br>"
        f"Initial capital={config.portfolio.initial_capital:,.2f}<br>Rebalancing Frequency={config.portfolio.rebalance_frequency_days} days"
        "<br>"
        f"Tolerance band={config.portfolio.tolerance_band:.2%}<br>Capital Gain Tax={config.portfolio.capital_gains_tax_rate:.2%}"
    )

    fig_spaghetti = plot_spaghetti_paths(
        wealth_paths=wealth_paths,
        n_sample=100,
        seed=42,
        normalize_to_1=False,
        title="Monte Carlo Spaghetti",
        subtitle=subtitle,
        bottom_note=summary_note,
        subtitle_align="left",
        bottom_note_align="left",
        bottom_note_x=0.65,
        bottom_note_y=-0.10,
        bottom_note_box=True,
        backend="plotly",
        width=800,
        height=600,
    )
    fig_spaghetti.update_xaxes(title_text="")

    fig_terminal = plot_terminal_wealth_distribution(
        wealth_paths=wealth_paths,
        bins=60,
        title="Distribution of Terminal Wealth",
        backend="plotly",
        width=800,
        height=600,
    )

    terminal_footnote = (
        f"min={terminal_summary['min']:,.0f} | p5={terminal_summary['p5']:,.0f} | "
        f"median={terminal_summary['median']:,.0f} | mean={terminal_summary['mean']:,.0f} | "
        f"p95={terminal_summary['p95']:,.0f} | max={terminal_summary['max']:,.0f}"
    )
    fig_combined = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.28,
        row_heights=[0.5, 0.5],
    )

    for tr in fig_spaghetti.data:
        fig_combined.add_trace(tr, row=1, col=1)

    for tr in fig_terminal.data:
        fig_combined.add_trace(tr, row=2, col=1)

    fig_combined.update_layout(
        width=900,
        height=1050,
        margin={"t": 110, "b": 100, "l": 70, "r": 30},
        showlegend=fig_spaghetti.layout.showlegend if fig_spaghetti.layout.showlegend is not None else False,
        font={"family": "Satoshi, 'Satoshi Variable', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "size": 12, "color": "#0f172a"},
        title_font={"family": "Satoshi"},
    )

    fig_combined.update_xaxes(title_text="", row=1, col=1)
    fig_combined.update_yaxes(
        title_text=fig_spaghetti.layout.yaxis.title.text if fig_spaghetti.layout.yaxis.title is not None else "Portfolio Value",
        row=1,
        col=1,
        showgrid=fig_spaghetti.layout.yaxis.showgrid if fig_spaghetti.layout.yaxis.showgrid is not None else True,
        gridcolor=fig_spaghetti.layout.yaxis.gridcolor if fig_spaghetti.layout.yaxis.gridcolor is not None else None,
    )

    fig_combined.update_xaxes(
        title_text=fig_terminal.layout.xaxis.title.text if fig_terminal.layout.xaxis.title is not None else "Terminal Wealth",
        row=2,
        col=1,
        showgrid=fig_terminal.layout.xaxis.showgrid if fig_terminal.layout.xaxis.showgrid is not None else True,
        gridcolor=fig_terminal.layout.xaxis.gridcolor if fig_terminal.layout.xaxis.gridcolor is not None else None,
    )
    fig_combined.update_yaxes(
        title_text=fig_terminal.layout.yaxis.title.text if fig_terminal.layout.yaxis.title is not None else "Frequency",
        row=2,
        col=1,
        showgrid=fig_terminal.layout.yaxis.showgrid if fig_terminal.layout.yaxis.showgrid is not None else True,
        gridcolor=fig_terminal.layout.yaxis.gridcolor if fig_terminal.layout.yaxis.gridcolor is not None else None,
    )

    fig_combined.add_annotation(
        x=0.0,
        y=1.12,
        xref="paper",
        yref="paper",
        text="Monte Carlo Spaghetti",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 20, "color": "#0f172a"},
    )

    fig_combined.add_annotation(
        x=0.0,
        y=1.07,
        xref="paper",
        yref="paper",
        text=subtitle,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 12, "color": "#334155"},
    )

    summary_box_y = 0.59
    metrics_box_y = 0.59

    fig_combined.add_annotation(
        x=0.65,
        y=summary_box_y,
        xref="paper",
        yref="paper",
        text=summary_note,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#334155"},
        bgcolor="rgba(255, 255, 255, 0.82)",
        bordercolor="#cbd5e1",
        borderwidth=1,
    )

    metrics_summary = result.metrics.summary
    if isinstance(metrics_summary, pd.DataFrame):
        if metrics_summary.shape[1] == 1:
            metrics_series = metrics_summary.iloc[:, 0]
        else:
            metrics_series = metrics_summary.mean(axis=1)
    else:
        metrics_series = pd.Series(metrics_summary)

    def _fmt_metric_value(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            if abs(v) >= 1000:
                return f"{v:,.2f}"
            return f"{v:.4f}"
        return str(v)

    metrics_table_df = pd.DataFrame(
        {
            "Metric": [str(idx) for idx in metrics_series.index],
            "Value": [_fmt_metric_value(v) for v in metrics_series.values],
        }
    )

    n_metrics = len(metrics_table_df)
    n_rows = max(1, int(np.ceil(n_metrics / 2)))
    left_metrics = metrics_table_df.iloc[:n_rows].reset_index(drop=True)
    right_metrics = metrics_table_df.iloc[n_rows:].reset_index(drop=True)
    if len(right_metrics) < n_rows:
        right_metrics = pd.concat(
            [
                right_metrics,
                pd.DataFrame(
                    {
                        "Metric": [""] * (n_rows - len(right_metrics)),
                        "Value": [""] * (n_rows - len(right_metrics)),
                    }
                ),
            ],
            ignore_index=True,
        )

    table_top = metrics_box_y
    table_height = 0.13
    table_left = 0.00
    table_width = 0.58

    row_fill = ["#ffffff" if i % 2 == 0 else "#f8fafc" for i in range(n_rows)]

    fig_combined.add_trace(
        go.Table(
            columnwidth=[0.34, 0.16, 0.34, 0.16],
            header={
                "values": ["<b>Metric</b>", "<b>Value</b>", "<b>Metric</b>", "<b>Value</b>"],
                "align": ["left", "right", "left", "right"],
                "fill_color": "#e2e8f0",
                "font": {"family": "Satoshi", "size": 11, "color": "#0f172a"},
                "line_color": "#cbd5e1",
                "height": 24,
            },
            cells={
                "values": [
                    left_metrics["Metric"],
                    left_metrics["Value"],
                    right_metrics["Metric"],
                    right_metrics["Value"],
                ],
                "align": ["left", "right", "left", "right"],
                "fill_color": [row_fill, row_fill, row_fill, row_fill],
                "font": {"family": "Satoshi", "size": 10, "color": "#0f172a"},
                "line_color": "#e2e8f0",
                "height": 21,
            },
            domain={
                "x": [table_left, table_left + table_width],
                "y": [table_top - table_height, table_top],
            },
        )
    )

    fig_combined.add_annotation(
        x=0.0,
        y=0.40,
        xref="paper",
        yref="paper",
        text="Distribution of Terminal Wealth",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 18, "color": "#0f172a"},
    )

    fig_combined.add_annotation(
        x=0.1,
        y=-0.05,
        xref="paper",
        yref="paper",
        text=terminal_footnote,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font={"size": 11, "color": "#334155"},
    )

    for fig in (fig_spaghetti, fig_terminal, fig_combined):
        fig.update_layout(
            font={"family": "Satoshi, 'Satoshi Variable', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", "size": 12, "color": "#0f172a"},
            title_font={"family": "Satoshi"},
        )
        fig.update_xaxes(tickfont={"family": "Satoshi"}, title_font={"family": "Satoshi"})
        fig.update_yaxes(tickfont={"family": "Satoshi"}, title_font={"family": "Satoshi"})

    if fig_combined.layout.annotations:
        for ann in fig_combined.layout.annotations:
            existing_font = ann.font.to_plotly_json() if ann.font else {}
            ann.font = {**existing_font, "family": "Satoshi"}

    summary_png_path = output_dir / "mc-simulation-summary.png"
    fig_combined.write_image(summary_png_path, format="png", width=900, height=1050, scale=2)


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
        _cleanup_legacy_exports(per_portfolio_dir)
        save_portfolio_metrics_summary(
            config=config,
            metrics_summary=result.metrics.summary,
            output_csv_path=per_portfolio_csv,
        )
        _set_portfolio_name(per_portfolio_csv, config, name)

        # 3) Save one combined summary figure in each portfolio output folder.
        _export_figures(
            portfolio_name=name,
            config=config,
            result=result,
            output_dir=per_portfolio_dir,
        )

        print(f"    Saved: {AGGREGATE_CSV}")
        print(f"    Saved: {per_portfolio_csv}")
        print(f"    Saved summary figure under: {per_portfolio_dir}")


if __name__ == "__main__":
    run_batch()