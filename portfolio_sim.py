from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from constants import TRADING_DAYS_PER_YEAR


@dataclass
class SimulationResult:
    wealth_paths: np.ndarray
    portfolio_returns: np.ndarray
    taxes_paid: np.ndarray
    rebalance_flags: np.ndarray


def _validate_weights(target_weights: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    if not target_weights:
        raise ValueError("target_weights cannot be empty.")

    asset_names = list(target_weights.keys())
    weights = np.asarray([float(target_weights[a]) for a in asset_names], dtype=float)

    if np.any(weights < 0):
        raise ValueError("target_weights must be non-negative.")
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Sum of target weights must be positive.")

    weights = weights / weight_sum
    return asset_names, weights


def _validate_simulated_returns(simulated_returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(simulated_returns, dtype=float)
    if arr.ndim != 3:
        raise ValueError("simulated_returns must have shape (paths, time, assets).")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("simulated_returns dimensions must all be > 0.")
    if not np.isfinite(arr).all():
        raise ValueError("simulated_returns contains NaN or inf values.")
    return arr


def simulate_portfolio_paths(
    simulated_returns: np.ndarray,
    target_weights: Mapping[str, float],
    initial_capital: float = 100_000.0,
    rebalance_frequency_days: int = TRADING_DAYS_PER_YEAR,
    tolerance_band: float = 0.05,
    capital_gains_tax_rate: float = 0.0,
) -> SimulationResult:
    """
    Simulate portfolio evolution path-by-path and day-by-day with exact PMC tax logic.

    Rebalance policy:
    - Check annually (default every 252 trading days).
    - If any current weight deviates from target by more than tolerance_band:
      rebalance whole portfolio back to targets.
    - Mandatory order: sell -> pay taxes -> buy.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0.")
    if rebalance_frequency_days <= 0:
        raise ValueError("rebalance_frequency_days must be > 0.")
    if tolerance_band < 0:
        raise ValueError("tolerance_band must be >= 0.")
    if not (0.0 <= capital_gains_tax_rate <= 1.0):
        raise ValueError("capital_gains_tax_rate must be in [0, 1].")

    returns = _validate_simulated_returns(simulated_returns)
    asset_names, w_target = _validate_weights(target_weights)

    n_paths, n_days, n_assets = returns.shape
    if n_assets != len(asset_names):
        raise ValueError(
            "Asset dimension mismatch: simulated_returns has "
            f"{n_assets} assets, target_weights has {len(asset_names)}."
        )

    wealth_paths = np.zeros((n_paths, n_days + 1), dtype=float)
    taxes_paid = np.zeros((n_paths, n_days), dtype=float)
    rebalance_flags = np.zeros((n_paths, n_days), dtype=bool)

    for p in range(n_paths):
        prices = np.ones(n_assets, dtype=float)
        shares = (initial_capital * w_target) / prices
        pmc = prices.copy()
        cash = 0.0

        wealth_paths[p, 0] = initial_capital

        for t in range(n_days):
            daily_r = returns[p, t, :]
            prices *= (1.0 + daily_r)

            asset_values = shares * prices
            total_value = float(asset_values.sum() + cash)
            wealth_paths[p, t + 1] = total_value

            if (t + 1) % rebalance_frequency_days != 0:
                continue
            if total_value <= 0.0:
                continue

            current_weights = asset_values / total_value
            drift = np.abs(current_weights - w_target)
            if not np.any(drift > tolerance_band):
                continue

            rebalance_flags[p, t] = True

            target_values = total_value * w_target
            value_diff = target_values - asset_values

            # 1) SELL
            realized_positive_gain = 0.0
            sell_mask = value_diff < 0.0

            for i in np.where(sell_mask)[0]:
                sell_value = min(-value_diff[i], shares[i] * prices[i])
                if sell_value <= 0.0 or prices[i] <= 0.0:
                    continue

                q_sell = sell_value / prices[i]
                gain = (prices[i] - pmc[i]) * q_sell

                shares[i] -= q_sell
                cash += sell_value

                if gain > 0.0:
                    realized_positive_gain += gain

            # 2) PAY TAXES
            tax_due = realized_positive_gain * capital_gains_tax_rate
            taxes_paid[p, t] = tax_due
            cash -= tax_due

            # 3) BUY
            post_sell_values = shares * prices
            post_sell_total = float(post_sell_values.sum() + cash)
            if post_sell_total <= 0.0:
                continue

            post_sell_targets = post_sell_total * w_target
            buy_values = np.maximum(post_sell_targets - post_sell_values, 0.0)

            for i in np.where(buy_values > 0.0)[0]:
                if cash <= 0.0 or prices[i] <= 0.0:
                    break

                buy_value = min(buy_values[i], cash)
                if buy_value <= 0.0:
                    continue

                q_old = shares[i]
                q_buy = buy_value / prices[i]
                q_new = q_old + q_buy

                if q_new > 0.0:
                    if q_old <= 0.0:
                        pmc[i] = prices[i]
                    else:
                        pmc[i] = ((pmc[i] * q_old) + (prices[i] * q_buy)) / q_new

                shares[i] = q_new
                cash -= buy_value

            if abs(cash) < 1e-12:
                cash = 0.0

            wealth_paths[p, t + 1] = float((shares * prices).sum() + cash)

    portfolio_returns = (wealth_paths[:, 1:] / wealth_paths[:, :-1]) - 1.0

    return SimulationResult(
        wealth_paths=wealth_paths,
        portfolio_returns=portfolio_returns,
        taxes_paid=taxes_paid,
        rebalance_flags=rebalance_flags,
    )