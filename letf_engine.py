"""Synthetic LETF daily return calculations.

This module builds leveraged ETF daily returns from an underlying return series
and financing costs, following the project specification formula.
"""

from __future__ import annotations

from numbers import Real

import pandas as pd
from constants import TRADING_DAYS_PER_YEAR


def _validate_scalar_rate(name: str, value: float) -> float:
    """Validate scalar rate-like input."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    return float(value)


def _coerce_borrowing_rate_to_series(
    borrowing_rate: pd.Series | float,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Return a trading-day indexed borrowing-rate series."""
    if isinstance(borrowing_rate, pd.Series):
        if borrowing_rate.empty:
            raise ValueError("borrowing_rate series is empty.")

        rate = borrowing_rate.copy()
        rate.index = pd.DatetimeIndex(rate.index)
        rate = rate.sort_index().reindex(index).ffill()

        if rate.isna().any():
            raise ValueError(
                "borrowing_rate cannot be aligned to all trading days. "
                "Provide wider rate history or pre-aligned data."
            )
        return rate.astype(float)

    if isinstance(borrowing_rate, Real):
        return pd.Series(float(borrowing_rate), index=index, name="borrowing_rate")

    raise TypeError("borrowing_rate must be a pandas Series or a float scalar.")


def synthetic_letf_daily_returns(
    underlying_returns: pd.Series,
    leverage: float,
    ter: float,
    borrowing_rate: pd.Series | float,
    spread: float = 0.0,
    *,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    borrowing_rate_is_annual: bool = False,
) -> pd.Series:
    """Compute synthetic LETF daily returns.

    The specification formula is:

    R_L,t = L * R_i,t - [TER / 252 + (L - 1) * (Rate_t + Spread) / 252]

    Parameters
    ----------
    underlying_returns:
        Trading-day indexed underlying daily simple returns.
    leverage:
        Leverage factor L (e.g., 2.0, 3.0).
    ter:
        Annual TER in decimal form (e.g., 0.0095 for 0.95%).
    borrowing_rate:
        Borrowing rate series or scalar in decimal form. By default this is
        interpreted as daily decimal rate because data_loader returns daily
        rates. Set ``borrowing_rate_is_annual=True`` if you pass annualized
        decimal rates.
    spread:
        Annual spread in decimal form.
    trading_days:
        Number of trading days used for annual-to-daily conversion.
    borrowing_rate_is_annual:
        If True, ``borrowing_rate`` is interpreted as annualized decimal rates
        and used directly in the specification formula. If False, rates are
        annualized internally before applying the same formula.

    Returns
    -------
    pd.Series
        Synthetic LETF daily returns indexed on trading days.
    """
    if not isinstance(underlying_returns, pd.Series):
        raise TypeError("underlying_returns must be a pandas Series.")
    if underlying_returns.empty:
        raise ValueError("underlying_returns is empty.")

    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    leverage = _validate_scalar_rate("leverage", leverage)
    ter = _validate_scalar_rate("ter", ter)
    spread = _validate_scalar_rate("spread", spread)

    if leverage <= 0:
        raise ValueError("leverage must be greater than zero.")

    returns = underlying_returns.astype(float).copy()
    returns.index = pd.DatetimeIndex(returns.index)
    returns = returns.sort_index()

    rate_series = _coerce_borrowing_rate_to_series(borrowing_rate, returns.index)

    # Keep one formula implementation: convert to annualized if inputs are daily.
    if borrowing_rate_is_annual:
        annual_rate = rate_series
    else:
        annual_rate = rate_series * float(trading_days)

    financing_cost = (ter / trading_days) + (leverage - 1.0) * (
        (annual_rate + spread) / trading_days
    )

    letf_returns = leverage * returns - financing_cost
    letf_returns.name = f"LETF_{leverage:g}x"
    return letf_returns
