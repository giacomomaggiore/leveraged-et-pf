from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Union

import numpy as np
import pandas as pd

from data_loader import load_market_data
from letf_engine import synthetic_letf_daily_returns
from metrics import MetricsResult, evaluate_metrics_from_simulation_result
from montecarlo import simulate_monte_carlo
from portfolio_sim import SimulationResult, simulate_portfolio_paths


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


@dataclass(frozen=True)
class SimulationConfig:
    """Single object passed from main.ipynb to run the full simulation."""

    market: MarketDataConfig
    assets: list[AssetConfig]
    portfolio: PortfolioConfig
    monte_carlo: MonteCarloConfig
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
    if not assets:
        raise ValueError("assets cannot be empty.")

    ids = [a.id for a in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("Each asset id must be unique.")


def _validate_target_weights(target_weights: Mapping[str, float], valid_asset_ids: set[str]) -> None:
    if not target_weights:
        raise ValueError("portfolio.target_weights cannot be empty.")

    missing = set(target_weights.keys()) - valid_asset_ids
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"target_weights contains unknown asset ids: {missing_str}")


def _build_historical_asset_returns(config: SimulationConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Create the historical return panel used as Monte Carlo input."""
    _validate_assets(config.assets)

    base_tickers: set[str] = set()
    for asset in config.assets:
        if isinstance(asset, SpotAssetConfig):
            base_tickers.add(asset.ticker)
        elif isinstance(asset, SyntheticLETFAssetConfig):
            base_tickers.add(asset.underlying_ticker)

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

    base_returns = base_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    daily_rate = daily_rate.reindex(base_returns.index).ffill()

    if base_returns.empty:
        raise ValueError(
            "No overlapping asset-return/rate history after alignment. "
            "Use a different rate series or a later start date."
        )

    asset_return_series: dict[str, pd.Series] = {}
    for asset in config.assets:
        if isinstance(asset, SpotAssetConfig):
            if asset.ticker not in base_returns.columns:
                raise ValueError(f"Missing returns for spot ticker '{asset.ticker}'.")
            s = base_returns[asset.ticker].copy()
            s.name = asset.id
            asset_return_series[asset.id] = s
            continue

        if asset.underlying_ticker not in base_returns.columns:
            raise ValueError(
                f"Missing returns for LETF underlying ticker '{asset.underlying_ticker}'."
            )

        letf_returns = synthetic_letf_daily_returns(
            underlying_returns=base_returns[asset.underlying_ticker],
            leverage=asset.leverage,
            ter=asset.ter,
            borrowing_rate=daily_rate,
            spread=asset.spread,
            borrowing_rate_is_annual=False,
        )
        letf_returns.name = asset.id
        asset_return_series[asset.id] = letf_returns

    historical_asset_returns = pd.concat(asset_return_series.values(), axis=1)
    historical_asset_returns = historical_asset_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")

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
    asset_ids = {asset.id for asset in config.assets}
    _validate_target_weights(config.portfolio.target_weights, valid_asset_ids=asset_ids)

    historical_asset_returns, daily_rate = _build_historical_asset_returns(config)

    simulated = simulate_monte_carlo(
        historical_returns=historical_asset_returns,
        n_paths=config.monte_carlo.n_paths,
        horizon_days=config.monte_carlo.horizon_days,
        method=config.monte_carlo.method,
        distribution=config.monte_carlo.distribution,
        student_t_df=config.monte_carlo.student_t_df,
        seed=config.monte_carlo.seed,
    )

    ordered_weights = {
        col: float(config.portfolio.target_weights[col])
        for col in historical_asset_returns.columns
        if col in config.portfolio.target_weights
    }

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
        risk_free_for_metrics = float(daily_rate.mean())
    else:
        risk_free_for_metrics = None

    metrics_result = evaluate_metrics_from_simulation_result(
        simulation_result=portfolio_result,
        risk_free_daily=risk_free_for_metrics,
        ruin_threshold_fraction=config.metrics_ruin_threshold_fraction,
    )

    return CompleteSimulationResult(
        historical_asset_returns=historical_asset_returns,
        aligned_daily_rate=daily_rate,
        simulated_asset_returns=simulated,
        portfolio=portfolio_result,
        metrics=metrics_result,
    )
