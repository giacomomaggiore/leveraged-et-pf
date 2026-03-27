"""Data acquisition and alignment utilities for the LETF simulator.

This module provides:
- Daily adjusted close downloads from yfinance
- Daily simple return computation
- FRED rate downloads via pandas-datareader
- Annual-to-daily rate conversion (252 trading days)
- Trading-day alignment between asset returns and rates
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

TRADING_DAYS_PER_YEAR = 252


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
	"""Validate and normalize a tickers iterable into a list of symbols."""
	normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
	if not normalized:
		raise ValueError("At least one valid ticker must be provided.")
	return normalized


def download_adj_close_prices(
	tickers: Iterable[str],
	start: str | datetime,
	end: str | datetime,
) -> pd.DataFrame:
	"""Download adjusted close prices for one or more tickers.

	Parameters
	----------
	tickers:
		Iterable of ticker symbols (e.g., ["SPY", "TLT"]).
	start, end:
		Date bounds accepted by yfinance.

	Returns
	-------
	pd.DataFrame
		Trading-day indexed DataFrame with one column per ticker.
	"""
	symbols = _normalize_tickers(tickers)
	raw = yf.download(
		tickers=symbols,
		start=start,
		end=end,
		auto_adjust=False,
		progress=False,
	)

	if raw.empty:
		raise ValueError("No price data returned by yfinance for requested inputs.")

	if "Adj Close" not in raw.columns:
		raise ValueError("yfinance response missing 'Adj Close' column.")

	adj_close = raw["Adj Close"]
	if isinstance(adj_close, pd.Series):
		adj_close = adj_close.to_frame(name=symbols[0])

	adj_close.index = pd.DatetimeIndex(adj_close.index)
	adj_close = adj_close.sort_index()
	return adj_close


def compute_daily_simple_returns(prices: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
	"""Compute daily simple returns from adjusted close prices."""
	if prices.empty:
		raise ValueError("Prices DataFrame is empty.")

	returns = prices.pct_change()
	returns = returns.replace([float("inf"), float("-inf")], pd.NA)
	if dropna:
		returns = returns.dropna(how="all")
	return returns


def fetch_fred_annual_rate(
	fred_series: str,
	start: str | datetime,
	end: str | datetime,
) -> pd.Series:
	"""Fetch a FRED annualized rate time series.

	Examples of series ids: 'SOFR', 'DFF', 'DTB3'.
	"""
	data = web.DataReader(fred_series, "fred", start, end)
	if data.empty:
		raise ValueError(f"No FRED data returned for series '{fred_series}'.")

	series = data.iloc[:, 0]
	series.name = fred_series
	series.index = pd.DatetimeIndex(series.index)
	series = series.sort_index()
	return series


def annual_to_daily_rate(
	annual_rate: pd.Series,
	trading_days: int = TRADING_DAYS_PER_YEAR,
	is_percent: bool = True,
) -> pd.Series:
	"""Convert annualized rate to daily decimal rate using trading-day convention.

	By default, FRED rates are interpreted as percent values (e.g., 4.5 means 4.5%).
	"""
	if annual_rate.empty:
		raise ValueError("Annual rate series is empty.")
	if trading_days <= 0:
		raise ValueError("trading_days must be a positive integer.")

	annual_decimal = annual_rate / 100.0 if is_percent else annual_rate
	daily_rate = annual_decimal / trading_days
	daily_rate.name = f"{annual_rate.name}_daily"
	return daily_rate


def align_returns_and_daily_rate(
	asset_returns: pd.DataFrame,
	daily_rate: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Align returns and rates on a trading-days-only index.

	The trading-day index is defined by asset returns. The rate series is reindexed
	to trading days and forward-filled to cover missing observations.
	"""
	if asset_returns.empty:
		raise ValueError("asset_returns is empty.")
	if daily_rate.empty:
		raise ValueError("daily_rate is empty.")

	trading_index = pd.DatetimeIndex(asset_returns.index).sort_values()

	aligned_returns = asset_returns.copy().sort_index()
	aligned_returns = aligned_returns.reindex(trading_index)

	aligned_rate = daily_rate.sort_index().reindex(trading_index).ffill()
	aligned_rate.name = daily_rate.name

	if aligned_rate.isna().all():
		raise ValueError("Rate series has no valid data after trading-day alignment.")

	return aligned_returns, aligned_rate


def load_market_data(
	tickers: Iterable[str],
	fred_series: str,
	start: str | datetime,
	end: str | datetime,
	fred_is_percent: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
	"""End-to-end loader returning aligned asset returns and daily rate.

	Output is aligned on trading days only and uses a 252-day annual-to-daily
	convention for rates.
	"""
	prices = download_adj_close_prices(tickers=tickers, start=start, end=end)
	returns = compute_daily_simple_returns(prices)

	annual_rate = fetch_fred_annual_rate(fred_series=fred_series, start=start, end=end)
	daily_rate = annual_to_daily_rate(annual_rate, trading_days=TRADING_DAYS_PER_YEAR, is_percent=fred_is_percent)

	return align_returns_and_daily_rate(asset_returns=returns, daily_rate=daily_rate)

