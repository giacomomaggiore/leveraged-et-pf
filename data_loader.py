"""Data acquisition and alignment utilities for the LETF simulator.

This module provides:
- Daily adjusted close downloads from yfinance
- Daily simple return computation
- FRED rate downloads via pandas-datareader
- Annual-to-daily rate conversion (252 trading days)
- Trading-day alignment between asset returns and rates
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

TRADING_DAYS_PER_YEAR = 252
YF_MAX_RETRIES = 4
YF_BASE_BACKOFF_SECONDS = 1.0
PRICE_CACHE_DIR = Path(__ggggfile__).resolve().parent / "output" / "price_cache"


def _is_rate_limit_error(exc: Exception) -> bool:
	"""Return True when an exception likely indicates a Yahoo rate limit."""
	name = exc.__class__.__name__.lower()
	text = str(exc).lower()
	return "ratelimit" in name or "too many requests" in text or "rate limited" in text


def _yf_download_with_retries(
	tickers: list[str],
	start: str | datetime,
	end: str | datetime,
	max_retries: int = YF_MAX_RETRIES,
	base_backoff_seconds: float = YF_BASE_BACKOFF_SECONDS,
) -> pd.DataFrame:
	"""Download from yfinance with bounded retries for transient failures."""
	last_exc: Exception | None = None

	for attempt in range(1, max_retries + 1):
		try:
			return yf.download(
				tickers=tickers,
				start=start,
				end=end,
				auto_adjust=False,
				progress=False,
				threads=False,
			)
		except Exception as exc:  # pragma: no cover - depends on upstream API behavior
			last_exc = exc
			if not _is_rate_limit_error(exc) or attempt == max_retries:
				raise

			sleep_s = base_backoff_seconds * (2 ** (attempt - 1))
			warnings.warn(
				(
					f"yfinance rate-limited for {tickers}. "
					f"Retry {attempt}/{max_retries} after {sleep_s:.1f}s."
				),
				RuntimeWarning,
			)
			time.sleep(sleep_s)

	if last_exc is not None:
		raise last_exc

	return pd.DataFrame()


def _extract_adj_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
	"""Normalize yfinance output to a ticker-column adjusted-close DataFrame."""
	if raw.empty:
		return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

	if isinstance(raw.columns, pd.MultiIndex):
		if "Adj Close" not in raw.columns.get_level_values(0):
			raise ValueError("yfinance response missing 'Adj Close' column.")
		adj_close = raw["Adj Close"]
	else:
		if "Adj Close" not in raw.columns:
			raise ValueError("yfinance response missing 'Adj Close' column.")
		adj_close = raw["Adj Close"]

	if isinstance(adj_close, pd.Series):
		adj_close = adj_close.to_frame(name=symbols[0])

	adj_close.columns = [str(col).strip().upper() for col in adj_close.columns]
	adj_close.index = pd.DatetimeIndex(adj_close.index)
	adj_close = adj_close.sort_index()
	return adj_close


def _cache_path_for_symbol(symbol: str) -> Path:
	"""Return the CSV path used to cache one symbol's price history."""
	safe = symbol.replace("=", "_EQ_").replace("/", "_")
	return PRICE_CACHE_DIR / f"{safe}.csv"


def _load_cached_symbol_prices(
	symbol: str,
	start: str | datetime,
	end: str | datetime,
) -> pd.Series | None:
	"""Load cached symbol prices if they fully cover the requested date range."""
	cache_path = _cache_path_for_symbol(symbol)
	if not cache_path.exists():
		return None

	try:
		cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
	except Exception:
		return None

	if cached.empty or "Adj Close" not in cached.columns:
		return None

	series = cached["Adj Close"].dropna().sort_index()
	if series.empty:
		return None

	start_ts = pd.Timestamp(start)
	end_ts = pd.Timestamp(end)
	if series.index.min() > start_ts or series.index.max() < end_ts:
		return None

	return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]


def _save_cached_symbol_prices(symbol: str, series: pd.Series) -> None:
	"""Persist symbol prices to a local CSV cache for offline/retry usage."""
	if series.empty:
		return

	PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	frame = series.sort_index().to_frame(name="Adj Close")
	frame.to_csv(_cache_path_for_symbol(symbol))


def _stooq_candidates(symbol: str) -> list[str]:
	"""Return likely stooq ticker variants for a Yahoo ticker symbol."""
	s = symbol.strip().upper()
	if s.endswith(".MI"):
		base = s[:-3]
		return [f"{base}.IT", s]
	if "." not in s:
		return [f"{s}.US", s]
	return [s]


def _download_from_stooq(symbol: str, start: str | datetime, end: str | datetime) -> pd.Series | None:
	"""Try to fetch close prices from stooq for a single symbol."""
	for candidate in _stooq_candidates(symbol):
		try:
			data = web.DataReader(candidate, "stooq", start, end)
		except Exception:
			continue

		if data.empty or "Close" not in data.columns:
			continue

		series = data["Close"].dropna().sort_index()
		if series.empty:
			continue

		series.name = symbol
		return series

	return None


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

	cached_frames: list[pd.DataFrame] = []
	uncached_symbols: list[str] = []
	for symbol in symbols:
		cached_series = _load_cached_symbol_prices(symbol=symbol, start=start, end=end)
		if cached_series is None or cached_series.empty:
			uncached_symbols.append(symbol)
			continue

		cached_frames.append(cached_series.to_frame(name=symbol))

	if uncached_symbols:
		warnings.warn(
			f"Local cache miss for symbols: {uncached_symbols}. Attempting online providers.",
			RuntimeWarning,
		)

	batch_raw = _yf_download_with_retries(tickers=uncached_symbols, start=start, end=end) if uncached_symbols else pd.DataFrame()
	try:
		batch_adj_close = _extract_adj_close(batch_raw, symbols=uncached_symbols) if uncached_symbols else pd.DataFrame()
	except ValueError:
		batch_adj_close = pd.DataFrame()

	available_symbols = [
		s for s in uncached_symbols
		if s in batch_adj_close.columns and not batch_adj_close[s].dropna().empty
	]
	missing_symbols = [s for s in uncached_symbols if s not in available_symbols]

	if not missing_symbols and (not batch_adj_close.empty or cached_frames):
		parts = []
		if cached_frames:
			parts.extend(cached_frames)
		if not batch_adj_close.empty:
			parts.append(batch_adj_close[available_symbols])
		combined_fast = pd.concat(parts, axis=1) if parts else pd.DataFrame()
		return combined_fast[symbols].sort_index()

	if missing_symbols:
		warnings.warn(
			(
				"Batch yfinance download returned incomplete data. "
				f"Falling back to per-ticker requests for: {missing_symbols}"
			),
			RuntimeWarning,
		)

	fallback_frames: list[pd.DataFrame] = []
	for symbol in missing_symbols:
		try:
			single_raw = _yf_download_with_retries(tickers=[symbol], start=start, end=end)
			single_adj_close = _extract_adj_close(single_raw, symbols=[symbol])
			if single_adj_close.empty or symbol not in single_adj_close.columns:
				stooq_series = _download_from_stooq(symbol=symbol, start=start, end=end)
				if stooq_series is not None and not stooq_series.empty:
					warnings.warn(
						f"Recovered {symbol} via stooq after empty Yahoo response.",
						RuntimeWarning,
					)
					fallback_frames.append(stooq_series.to_frame(name=symbol))
					_save_cached_symbol_prices(symbol=symbol, series=stooq_series)
				continue
			fallback_frames.append(single_adj_close[[symbol]])
			_save_cached_symbol_prices(symbol=symbol, series=single_adj_close[symbol].dropna())
		except Exception as exc:  # pragma: no cover - depends on upstream API behavior
			if _is_rate_limit_error(exc):
				stooq_series = _download_from_stooq(symbol=symbol, start=start, end=end)
				if stooq_series is not None and not stooq_series.empty:
					warnings.warn(
						f"Recovered {symbol} via stooq after Yahoo rate limit.",
						RuntimeWarning,
					)
					fallback_frames.append(stooq_series.to_frame(name=symbol))
					_save_cached_symbol_prices(symbol=symbol, series=stooq_series)
					continue

				warnings.warn(f"Rate limited while fetching {symbol}: {exc}", RuntimeWarning)
				continue
			raise

	batch_part = batch_adj_close[available_symbols].copy() if available_symbols else pd.DataFrame()
	if not batch_part.empty:
		for symbol in available_symbols:
			_save_cached_symbol_prices(symbol=symbol, series=batch_part[symbol].dropna())

	combined_parts: list[pd.DataFrame] = []
	if cached_frames:
		combined_parts.extend(cached_frames)
	if not batch_part.empty:
		combined_parts.append(batch_part)

	combined = pd.concat(combined_parts, axis=1) if combined_parts else pd.DataFrame()
	if fallback_frames:
		combined = pd.concat([combined, *fallback_frames], axis=1)

	if combined.empty:
		raise ValueError("No price data returned by yfinance for requested inputs.")

	final_missing = [s for s in symbols if s not in combined.columns or combined[s].dropna().empty]
	if final_missing:
		raise ValueError(
			"Failed to download adjusted close prices for tickers: "
			f"{final_missing}. This can happen during Yahoo Finance rate limits."
		)

	combined = combined[symbols]
	combined.index = pd.DatetimeIndex(combined.index)
	combined = combined.sort_index()
	return combined


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

	#returns daily returns and daily rates to apply the leverage
	return align_returns_and_daily_rate(asset_returns=returns, daily_rate=daily_rate)

