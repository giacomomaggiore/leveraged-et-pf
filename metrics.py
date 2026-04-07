from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class MetricsResult:
    """Container for pathwise metrics and their aggregated summaries."""

    pathwise: pd.DataFrame
    summary: pd.DataFrame
    drawdowns: np.ndarray


def _validate_wealth_paths(wealth_paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(wealth_paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError("wealth_paths must have shape (paths, time+1).")
    if arr.shape[0] <= 0 or arr.shape[1] <= 1:
        raise ValueError("wealth_paths must have at least one path and two time points.")
    if not np.isfinite(arr).all():
        raise ValueError("wealth_paths contains NaN or inf values.")
    if np.any(arr <= 0.0):
        raise ValueError("wealth_paths must be strictly positive for return metrics.")
    return arr


def _portfolio_returns_from_wealth(wealth_paths: np.ndarray) -> np.ndarray:
    return (wealth_paths[:, 1:] / wealth_paths[:, :-1]) - 1.0


def _coerce_risk_free_daily(
    risk_free_daily: float | pd.Series | np.ndarray | None,
    n_paths: int,
    n_days: int,
) -> np.ndarray:
    """Return risk-free daily rates as (paths, days)."""
    if risk_free_daily is None:
        return np.zeros((n_paths, n_days), dtype=float)

    if isinstance(risk_free_daily, (int, float)):
        return np.full((n_paths, n_days), float(risk_free_daily), dtype=float)

    if isinstance(risk_free_daily, pd.Series):
        arr = risk_free_daily.to_numpy(dtype=float)
    else:
        arr = np.asarray(risk_free_daily, dtype=float)

    if arr.ndim == 1:
        if arr.shape[0] != n_days:
            raise ValueError(
                "1D risk_free_daily must have length equal to number of simulation days."
            )
        return np.broadcast_to(arr.reshape(1, -1), (n_paths, n_days)).copy()

    if arr.ndim == 2:
        if arr.shape != (n_paths, n_days):
            raise ValueError("2D risk_free_daily must have shape (paths, days).")
        return arr

    raise ValueError("risk_free_daily must be scalar, 1D/2D array, pandas Series, or None.")


def _compute_drawdowns(wealth_paths: np.ndarray) -> np.ndarray:
    high_water_mark = np.maximum.accumulate(wealth_paths, axis=1)
    return (wealth_paths / high_water_mark) - 1.0


def _max_drawdown(drawdowns: np.ndarray) -> np.ndarray:
    return drawdowns.min(axis=1)


def _max_drawdown_duration(drawdowns: np.ndarray) -> np.ndarray:
    """Maximum consecutive trading days below high-water mark for each path."""
    below_hwm = drawdowns < 0.0
    n_paths, n_cols = below_hwm.shape
    durations = np.zeros(n_paths, dtype=int)

    for p in range(n_paths):
        max_run = 0
        run = 0
        for t in range(1, n_cols):
            if below_hwm[p, t]:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        durations[p] = max_run

    return durations


def _cagr(wealth_paths: np.ndarray, trading_days: int) -> np.ndarray:
    n_days = wealth_paths.shape[1] - 1
    total_return = wealth_paths[:, -1] / wealth_paths[:, 0]
    return np.power(total_return, trading_days / n_days) - 1.0


def _sharpe_ratio(
    portfolio_returns: np.ndarray,
    risk_free_daily: np.ndarray,
    trading_days: int,
) -> np.ndarray:
    excess = portfolio_returns - risk_free_daily
    mean_excess = excess.mean(axis=1)
    std_excess = excess.std(axis=1, ddof=1)

    sharpe = np.full(excess.shape[0], np.nan, dtype=float)
    valid = std_excess > 0.0
    sharpe[valid] = (mean_excess[valid] / std_excess[valid]) * np.sqrt(trading_days)
    return sharpe


def _sortino_ratio(
    portfolio_returns: np.ndarray,
    risk_free_daily: np.ndarray,
    trading_days: int,
) -> np.ndarray:
    excess = portfolio_returns - risk_free_daily
    mean_excess = excess.mean(axis=1)

    n_paths = excess.shape[0]
    sortino = np.full(n_paths, np.nan, dtype=float)

    for p in range(n_paths):
        downside = excess[p, excess[p] < 0.0]
        if downside.size < 2:
            continue

        downside_std = downside.std(ddof=1)
        if downside_std > 0.0:
            sortino[p] = (mean_excess[p] / downside_std) * np.sqrt(trading_days)

    return sortino


def _ulcer_index(drawdowns: np.ndarray) -> np.ndarray:
    """Ulcer Index as RMS of percentage drawdowns below high-water mark."""
    dd_pct = (-np.minimum(drawdowns, 0.0)) * 100.0
    return np.sqrt(np.mean(np.square(dd_pct), axis=1))


def _probability_of_ruin(
    wealth_paths: np.ndarray,
    ruin_threshold_fraction: float,
) -> tuple[np.ndarray, float]:
    if not (0.0 < ruin_threshold_fraction < 1.0):
        raise ValueError("ruin_threshold_fraction must be in (0, 1).")

    ruin_level = wealth_paths[:, [0]] * ruin_threshold_fraction
    ruined = np.any(wealth_paths <= ruin_level, axis=1)
    prob = float(ruined.mean())
    return ruined, prob


def _summary_stats(pathwise: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "mean": pathwise.mean(axis=0, skipna=True),
            "median": pathwise.median(axis=0, skipna=True),
            "p5": pathwise.quantile(0.05, axis=0, interpolation="linear"),
            "p95": pathwise.quantile(0.95, axis=0, interpolation="linear"),
        }
    )
    stats.index.name = "metric"
    return stats


def evaluate_paths_metrics(
    wealth_paths: np.ndarray,
    risk_free_daily: float | pd.Series | np.ndarray | None = None,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    ruin_threshold_fraction: float = 0.10,
) -> MetricsResult:
    """Compute required risk/return metrics and aggregate distribution statistics.

    Parameters
    ----------
    wealth_paths:
        Portfolio value matrix with shape (paths, time+1).
    risk_free_daily:
        Daily risk-free rate as scalar, 1D array/Series (days), 2D array
        (paths, days), or None (assumed zero).
    trading_days:
        Annualization factor. Defaults to 252.
    ruin_threshold_fraction:
        Ruin threshold as fraction of initial capital. Default 0.10 means
        ruin occurs when wealth falls below 10% of initial value.

    Returns
    -------
    MetricsResult
        Pathwise metrics, summary table, and full drawdown matrix.
    """
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    wealth = _validate_wealth_paths(wealth_paths)
    n_paths, n_cols = wealth.shape
    n_days = n_cols - 1

    portfolio_returns = _portfolio_returns_from_wealth(wealth)
    rf_daily = _coerce_risk_free_daily(risk_free_daily, n_paths=n_paths, n_days=n_days)

    drawdowns = _compute_drawdowns(wealth)
    cagr = _cagr(wealth, trading_days=trading_days)
    max_dd = _max_drawdown(drawdowns)
    dd_duration = _max_drawdown_duration(drawdowns)
    sharpe = _sharpe_ratio(portfolio_returns, rf_daily, trading_days=trading_days)
    sortino = _sortino_ratio(portfolio_returns, rf_daily, trading_days=trading_days)
    ulcer = _ulcer_index(drawdowns)
    ruined, probability_of_ruin = _probability_of_ruin(
        wealth_paths=wealth,
        ruin_threshold_fraction=ruin_threshold_fraction,
    )

    pathwise = pd.DataFrame(
        {
            "CAGR": cagr,
            "Max Drawdown": max_dd,
            "Drawdown Duration (Days)": dd_duration,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Ulcer Index": ulcer,
            "Ruined": ruined.astype(int),
        }
    )

    summary = _summary_stats(pathwise)
    summary.loc["Probability Of Ruin"] = {
        "mean": probability_of_ruin,
        "median": probability_of_ruin,
        "p5": probability_of_ruin,
        "p95": probability_of_ruin,
    }

    return MetricsResult(pathwise=pathwise, summary=summary, drawdowns=drawdowns)


def evaluate_metrics_from_simulation_result(
    simulation_result: Any,
    risk_free_daily: float | pd.Series | np.ndarray | None = None,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    ruin_threshold_fraction: float = 0.10,
) -> MetricsResult:
    """Convenience wrapper for portfolio_sim.SimulationResult outputs."""
    if not hasattr(simulation_result, "wealth_paths"):
        raise TypeError("simulation_result must expose a 'wealth_paths' attribute.")

    return evaluate_paths_metrics(
        wealth_paths=np.asarray(simulation_result.wealth_paths, dtype=float),
        risk_free_daily=risk_free_daily,
        trading_days=trading_days,
        ruin_threshold_fraction=ruin_threshold_fraction,
    )
