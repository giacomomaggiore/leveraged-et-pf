from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Union

import numpy as np
import pandas as pd

from data_loader import load_market_data
from letf_engine import synthetic_letf_daily_returns
from metrics import MetricsResult, evaluate_metrics_from_simulation_result
from montecarlo import simulate_monte_carlo
from portfolio_sim import SimulationResult, simulate_portfolio_paths


def _canonical_portfolio_name(config: "SimulationConfig") -> str:
    """Build a stable portfolio identifier from ticker, weight, and leverage."""
    rows: list[tuple[str, float, float]] = []

    for asset in sorted(config.assets, key=lambda a: a.id):
        weight = float(config.portfolio.target_weights.get(asset.id, 0.0))

        if isinstance(asset, SpotAssetConfig):
            ticker = asset.ticker
            leverage = 1.0
        else:
            ticker = asset.underlying_ticker
            leverage = float(asset.leverage)

        rows.append((ticker, weight, leverage))

    return " | ".join(
        f"{ticker} w={weight:.6f} lev={leverage:.6f}"
        for ticker, weight, leverage in rows
    )


def _flatten_metrics_summary(metrics_summary: pd.DataFrame) -> dict[str, float]:
    """Flatten summary table to one CSV row with deterministic column names."""
    if not isinstance(metrics_summary, pd.DataFrame):
        raise TypeError("metrics_summary must be a pandas DataFrame.")

    out: dict[str, float] = {}
    for metric_name, row in metrics_summary.iterrows():
        metric = str(metric_name).strip().replace(" ", "_")
        for stat_name, value in row.items():
            stat = str(stat_name).strip().replace(" ", "_")
            key = f"{metric}__{stat}"
            out[key] = float(value) if pd.notna(value) else np.nan

    return out


def save_portfolio_metrics_summary(
    *,
    config: "SimulationConfig",
    metrics_summary: pd.DataFrame,
    output_csv_path: str | Path = "output/portfolio_metrics_summary.csv",
) -> Path:
    """Upsert one portfolio summary row in output CSV.

    If a row with the same canonical portfolio composition already exists,
    it is updated. Otherwise, a new row is appended.
    """
    portfolio_composition = _canonical_portfolio_name(config)
    row_payload = {
        "portfolio composition": portfolio_composition,
        **_flatten_metrics_summary(metrics_summary),
    }

    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([row_payload])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)

        # Backward compatibility with previous schema.
        if "portfolio_name" in existing.columns and "portfolio composition" not in existing.columns:
            existing = existing.rename(columns={"portfolio_name": "portfolio composition"})

        if "portfolio composition" not in existing.columns:
            raise ValueError(
                f"Existing CSV '{csv_path}' must contain a 'portfolio composition' column."
            )

        if "portfolio_name" not in existing.columns:
            existing["portfolio_name"] = ""

        # Keep existing column order and extend only when new metrics appear.
        all_columns = [
            "portfolio composition",
            "portfolio_name",
            *[
                c
                for c in existing.columns
                if c not in {"portfolio composition", "portfolio_name"}
            ],
        ]
        for col in new_row.columns:
            if col not in all_columns:
                all_columns.append(col)

        if "portfolio_name" not in new_row.columns:
            new_row["portfolio_name"] = ""

        existing = existing.reindex(columns=all_columns)
        new_row = new_row.reindex(columns=all_columns)

        match_mask = existing["portfolio composition"] == portfolio_composition
        if bool(match_mask.any()):
            # Do not overwrite user-managed portfolio_name labels.
            update_columns = [c for c in all_columns if c != "portfolio_name"]
            existing.loc[match_mask, update_columns] = new_row.iloc[0][update_columns].tolist()
            final_df = existing
        else:
            final_df = pd.concat([existing, new_row], ignore_index=True)
    else:
        final_df = new_row

    ordered_cols = [
        "portfolio composition",
        "portfolio_name",
        *[c for c in final_df.columns if c not in {"portfolio composition", "portfolio_name"}],
    ]
    final_df = final_df[ordered_cols]
    final_df.to_csv(csv_path, index=False)
    return csv_path

#dataclass definition
#datafrozen = readonly after creation
@dataclass(frozen=True)
class MarketDataConfig:
    """Inputs required to fetch and align historical market data."""

    start: str
    end: str
    fred_series: str = "SOFR"
    fred_is_percent: bool = True


@dataclass(frozen=True)
class SpotAssetConfig:
    """A non-leveraged asset directly mapped to a market ticker."""

    id: str
    ticker: str
    kind: Literal["spot_etf"] = "spot_etf"


#to define synthetic leveraged etf, you need to specify:
# ticker
# daily leverage ratio 
# spread 
# ter
# the id can be chosen freely as you like (also Banana ETF works)
@dataclass(frozen=True)
class SyntheticLETFAssetConfig:
    """A synthetic LETF built from underlying returns and financing costs."""


    id: str
    underlying_ticker: str
    leverage: float
    ter: float
    spread: float = 0.0
    kind: Literal["synthetic_letf"] = "synthetic_letf"


AssetConfig = Union[SpotAssetConfig, SyntheticLETFAssetConfig]


@dataclass(frozen=True)
class PortfolioConfig:
    """Target allocation for the portfolio simulator."""

    target_weights: Mapping[str, float]
    initial_capital: float = 100_000.0
    rebalance_frequency_days: int = 252
    tolerance_band: float = 0.05
    capital_gains_tax_rate: float = 0.0


@dataclass(frozen=True)
class MonteCarloConfig:
    """Monte Carlo path-generation settings."""

    n_paths: int
    horizon_days: int
    method: Literal["bootstrap", "parametric"] = "bootstrap"
    distribution: Literal["normal", "student_t"] = "normal"
    student_t_df: float = 6.0
    seed: int | None = None


#simulation config will be passed to the main.ipynb to run the full simulation,
#it contains all the necessary information 
@dataclass(frozen=True)
class SimulationConfig:
    """Single object passed from main.ipynb to run the full simulation."""

    market: MarketDataConfig
    assets: list[AssetConfig]
    portfolio: PortfolioConfig
    monte_carlo: MonteCarloConfig
    
    # portfolio is considered ruined if wealth falls below this fraction 
    # of initial capital at any point in the path
    metrics_ruin_threshold_fraction: float = 0.10
    use_mean_risk_free_for_metrics: bool = True


@dataclass
class CompleteSimulationResult:
    """Artifacts produced by one end-to-end simulation run."""

    historical_asset_returns: pd.DataFrame
    aligned_daily_rate: pd.Series
    simulated_asset_returns: np.ndarray
    portfolio: SimulationResult
    metrics: MetricsResult


def _validate_assets(assets: list[AssetConfig]) -> None:
    # Common notebook typo: writing `asset_info = [...],` creates a tuple(list).
    if isinstance(assets, tuple) and len(assets) == 1 and isinstance(assets[0], list):
        raise ValueError(
            "assets must be a list of AssetConfig, not a tuple containing a list. "
            "If this comes from `asset_info`, remove the trailing comma after `]`."
        )

    if not isinstance(assets, list):
        raise ValueError("assets must be a list of AssetConfig entries.")

    # Guard against an invalid simulation setup with no assets at all.
    if not assets:
        raise ValueError("assets cannot be empty.")

    invalid_idx = [
        i
        for i, asset in enumerate(assets)
        if not isinstance(asset, (SpotAssetConfig, SyntheticLETFAssetConfig))
    ]
    if invalid_idx:
        raise ValueError(
            f"assets contains invalid entries at positions: {invalid_idx}. "
            "Each entry must be SpotAssetConfig or SyntheticLETFAssetConfig."
        )

    # Every asset id is used as a unique key later, so duplicates are not allowed.
    ids = [a.id for a in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Each asset id must be unique.")


def _validate_target_weights(target_weights: Mapping[str, float], valid_asset_ids: set[str]) -> None:
    # Portfolio allocation must be explicitly provided.
    if not target_weights:
        raise ValueError("portfolio.target_weights cannot be empty.")

    # Ensure weights only reference assets that actually exist in config.assets.
    missing = set(target_weights.keys()) - valid_asset_ids
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"target_weights contains unknown asset ids: {missing_str}")


def _build_historical_asset_returns(config: SimulationConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Create the historical return panel used as Monte Carlo input."""
    #  validate that asset definitions are structurally sound.
    _validate_assets(config.assets)

    # Collect only the base market tickers needed to build all configured assets.
    base_tickers: set[str] = set()
    
    # insert in base_tickers the tickers of all spot assets 
    # and the underlying tickers of all synthetic LETF assets
    for asset in config.assets:
        if isinstance(asset, SpotAssetConfig):
            base_tickers.add(asset.ticker)
        elif isinstance(asset, SyntheticLETFAssetConfig):
            base_tickers.add(asset.underlying_ticker)
    
    print("Base tickers needed for historical data:")
    print(base_tickers)

    
    base_returns, daily_rate = load_market_data(
        tickers=sorted(base_tickers),
        fred_series=config.market.fred_series,
        start=config.market.start,
        end=config.market.end,
        fred_is_percent=config.market.fred_is_percent,
    )

    # Keep only the overlapping window where both returns and borrowing rate exist.
    # This avoids leading NaNs for rate series that start later than asset history
    # (e.g., SOFR starts in 2018 while SPY/TLT data may start much earlier).
    overlap_mask = daily_rate.notna()
    if not overlap_mask.any():
        raise ValueError(
            "Selected rate series has no valid observations in the requested date range."
        )

    
    base_returns = base_returns.loc[overlap_mask]
    daily_rate = daily_rate.loc[overlap_mask]

    # Remove invalid rows so downstream simulations receive clean numeric data.
    base_returns = base_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    daily_rate = daily_rate.reindex(base_returns.index).ffill()


    #check for empty
    if base_returns.empty:
        raise ValueError(
            "No overlapping asset-return/rate history after alignment. "
            "Use a different rate series or a later start date."
        )

    #empty dict 
    asset_return_series: dict[str, pd.Series] = {}
    for asset in config.assets:
        # spot asset - just rename the base return series to the asset id and add to the dict
        if isinstance(asset, SpotAssetConfig):
            if asset.ticker not in base_returns.columns:
                raise ValueError(f"Missing returns for spot ticker '{asset.ticker}'.")
            s = base_returns[asset.ticker].copy()
            s.name = asset.id
            asset_return_series[asset.id] = s
            # is spot etf= skip the following logic and goes to the second one
            continue

        if asset.underlying_ticker not in base_returns.columns:
            raise ValueError(
                f"Missing returns for LETF underlying ticker '{asset.underlying_ticker}'."
            )

        # Build synthetic LETF returns from underlying returns and financing costs.
        letf_returns = synthetic_letf_daily_returns(
            #call leveraged etf function to compute the daily leverage returns 
            underlying_returns=base_returns[asset.underlying_ticker],
            leverage=asset.leverage,
            ter=asset.ter,
            borrowing_rate=daily_rate,
            spread=asset.spread,
            borrowing_rate_is_annual=False,
        )
        letf_returns.name = asset.id
        asset_return_series[asset.id] = letf_returns


    #build historical returns asset with both spot and synthetic etf returns
    historical_asset_returns = pd.concat(asset_return_series.values(), axis=1)
    # Final cleanup in case synthetic series introduced any non-finite values.
    #replace inf values 
    historical_asset_returns = historical_asset_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


    #check! 
    if historical_asset_returns.empty:
        raise ValueError("No aligned historical returns available for selected assets.")

    daily_rate = daily_rate.reindex(historical_asset_returns.index).ffill()
    if daily_rate.isna().any():
        raise ValueError("Daily rate could not be aligned to historical asset returns.")

    return historical_asset_returns, daily_rate


def run_complete_simulation(config: SimulationConfig) -> CompleteSimulationResult:
    """Run data loading, LETF construction, Monte Carlo, portfolio simulation, and metrics.

    This is the one-call orchestration API intended for main.ipynb usage.
    """
    _validate_assets(config.assets)

    # Validate that portfolio weights reference known assets.
    asset_ids = {asset.id for asset in config.assets}
    _validate_target_weights(config.portfolio.target_weights, valid_asset_ids=asset_ids)

    # Build the clean historical dataset used as input for path generation.
    historical_asset_returns, daily_rate = _build_historical_asset_returns(config)

    # Generate simulated daily asset returns across all paths and horizon days.
    simulated = simulate_monte_carlo(
        historical_returns=historical_asset_returns,
        n_paths=config.monte_carlo.n_paths,
        horizon_days=config.monte_carlo.horizon_days,
        method=config.monte_carlo.method,
        distribution=config.monte_carlo.distribution,
        student_t_df=config.monte_carlo.student_t_df,
        seed=config.monte_carlo.seed,
    )

    # Reorder weights to match the exact column order expected by the simulator.
    ordered_weights = {
        col: float(config.portfolio.target_weights[col])
        for col in historical_asset_returns.columns
        if col in config.portfolio.target_weights
    }

    # Simulate the portfolio path, including rebalancing and optional tax drag.
    portfolio_result = simulate_portfolio_paths(
        simulated_returns=simulated,
        target_weights=ordered_weights,
        initial_capital=config.portfolio.initial_capital,
        rebalance_frequency_days=config.portfolio.rebalance_frequency_days,
        tolerance_band=config.portfolio.tolerance_band,
        capital_gains_tax_rate=config.portfolio.capital_gains_tax_rate,
    )

    risk_free_for_metrics: float | pd.Series | None
    if config.use_mean_risk_free_for_metrics:
        # Use one average daily risk-free value for all paths when computing ratios.
        risk_free_for_metrics = float(daily_rate.mean())
    else:
        # Let metrics module use its default behavior when no risk-free input is provided.
        risk_free_for_metrics = None

    # Compute performance and risk statistics from the simulated portfolio outcomes.
    metrics_result = evaluate_metrics_from_simulation_result(
        simulation_result=portfolio_result,
        risk_free_daily=risk_free_for_metrics,
        ruin_threshold_fraction=config.metrics_ruin_threshold_fraction,
    )

    # Return all intermediate and final artifacts for analysis and plotting.
    return CompleteSimulationResult(
        historical_asset_returns=historical_asset_returns,
        aligned_daily_rate=daily_rate,
        simulated_asset_returns=simulated,
        portfolio=portfolio_result,
        metrics=metrics_result,
    )
