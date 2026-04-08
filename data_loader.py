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
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

TRADING_DAYS_PER_YEAR = 252
YF_MAX_RETRIES = 6
YF_BASE_BACKOFF_SECONDS = 2.0
YF_JITTER_FRACTION = 0.35
YF_MAX_BACKOFF_SECONDS = 90.0
PRICE_CACHE_DIR = Path(__file__).resolve().parent / "data"


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

			exponential = base_backoff_seconds * (2 ** (attempt - 1))
			jitter = random.uniform(0.0, exponential * YF_JITTER_FRACTION)
			sleep_s = min(YF_MAX_BACKOFF_SECONDS, exponential + jitter)
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
	"""Load cached symbol prices for the requested range when overlap exists."""
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
	if series.index.max() < start_ts or series.index.min() > end_ts:
		return None

	overlap = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]
	if overlap.empty:
		return None

	return overlap


def _save_cached_symbol_prices(symbol: str, series: pd.Series) -> None:
	"""Persist symbol prices to a local CSV cache for offline/retry usage."""
	if series.empty:
		return

	PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache_path = _cache_path_for_symbol(symbol)
	new_series = series.dropna().sort_index()

	if cache_path.exists():
		try:
			existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
		except Exception:
			existing = pd.DataFrame()

		if not existing.empty and "Adj Close" in existing.columns:
			existing_series = existing["Adj Close"].dropna().sort_index()
			new_series = pd.concat([existing_series, new_series]).sort_index()
			new_series = new_series[~new_series.index.duplicated(keep="last")]

	frame = new_series.to_frame(name="Adj Close")
	frame.to_csv(cache_path)


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
	start_ts = pd.Timestamp(start)
	end_ts = pd.Timestamp(end)
	one_day = pd.Timedelta(days=1)
	yf_end_ts = end_ts + one_day

	cached_frames: list[pd.DataFrame] = []
	online_start_by_symbol: dict[str, pd.Timestamp] = {}
	online_reason_by_symbol: dict[str, str] = {}
	for symbol in symbols:
		cached_series = _load_cached_symbol_prices(symbol=symbol, start=start, end=end)
		if cached_series is None or cached_series.empty:
			online_start_by_symbol[symbol] = start_ts
			online_reason_by_symbol[symbol] = "cache_miss"
			continue

		cached_frames.append(cached_series.to_frame(name=symbol))

		cached_max = pd.Timestamp(cached_series.index.max())
		if cached_max < end_ts:
			refresh_start = cached_max + one_day
			if refresh_start <= end_ts:
				online_start_by_symbol[symbol] = refresh_start
				online_reason_by_symbol[symbol] = "cache_stale"
				warnings.warn(
					(
						f"Local cached {symbol} is stale at {cached_max.date()} "
						f"(requested end={end_ts.date()}); fetching incremental updates online."
					),
					RuntimeWarning,
				)

	online_symbols = list(online_start_by_symbol.keys())
	if online_symbols:
		missing_only = [s for s in online_symbols if online_reason_by_symbol.get(s) == "cache_miss"]
		refresh_only = [s for s in online_symbols if online_reason_by_symbol.get(s) != "cache_miss"]
		if missing_only:
			warnings.warn(
				f"Local cache miss for symbols: {missing_only}. Attempting online providers.",
				RuntimeWarning,
			)
		if refresh_only:
			warnings.warn(
				f"Local cache refresh required for symbols: {refresh_only}. Attempting online providers.",
				RuntimeWarning,
			)

	batch_start = min(online_start_by_symbol.values()) if online_symbols else start_ts

	batch_raw = _yf_download_with_retries(tickers=online_symbols, start=batch_start, end=yf_end_ts) if online_symbols else pd.DataFrame()
	try:
		batch_adj_close = _extract_adj_close(batch_raw, symbols=online_symbols) if online_symbols else pd.DataFrame()
	except ValueError:
		batch_adj_close = pd.DataFrame()

	batch_frames: list[pd.DataFrame] = []
	available_symbols: list[str] = []
	missing_symbols: list[str] = []
	for symbol in online_symbols:
		if symbol not in batch_adj_close.columns:
			missing_symbols.append(symbol)
			continue

		filtered = batch_adj_close[symbol].dropna()
		filtered = filtered.loc[filtered.index >= online_start_by_symbol[symbol]]
		if filtered.empty:
			missing_symbols.append(symbol)
			continue

		available_symbols.append(symbol)
		batch_frames.append(filtered.to_frame(name=symbol))

	likely_batch_provider_failure = bool(online_symbols) and not available_symbols and batch_adj_close.empty

	if not missing_symbols and (batch_frames or cached_frames):
		# Persist fresh batch data before returning the assembled frame.
		for frame in batch_frames:
			symbol = str(frame.columns[0])
			_save_cached_symbol_prices(symbol=symbol, series=frame[symbol].dropna())

		parts = []
		if cached_frames:
			parts.extend(cached_frames)
		if batch_frames:
			parts.extend(batch_frames)
		combined_fast = pd.concat(parts, axis=1) if parts else pd.DataFrame()
		combined_fast = combined_fast[~combined_fast.index.duplicated(keep="last")]
		return combined_fast[symbols].sort_index()

	if missing_symbols:
		warnings.warn(
			(
				"Batch yfinance download returned incomplete data. "
				f"Falling back to per-ticker recovery for: {missing_symbols}"
			),
			RuntimeWarning,
		)

	if likely_batch_provider_failure and missing_symbols:
		warnings.warn(
			(
				"Yahoo Finance batch response returned no usable rows for all uncached symbols. "
				"Likely temporary provider throttling; skipping repeated per-ticker Yahoo calls."
			),
			RuntimeWarning,
		)

	fallback_frames: list[pd.DataFrame] = []
	for symbol in missing_symbols:
		fetch_start = online_start_by_symbol[symbol]
		if likely_batch_provider_failure:
			stooq_series = _download_from_stooq(symbol=symbol, start=fetch_start, end=end_ts)
			if stooq_series is not None and not stooq_series.empty:
				warnings.warn(
					f"Recovered {symbol} via stooq after Yahoo batch failure.",
					RuntimeWarning,
				)
				fallback_frames.append(stooq_series.to_frame(name=symbol))
				_save_cached_symbol_prices(symbol=symbol, series=stooq_series)
				continue

			warnings.warn(
				(
					f"No fallback data source could recover {symbol} after Yahoo batch failure. "
					"Try again later or provide a local cache CSV in data/."
				),
				RuntimeWarning,
			)
			continue

		try:
			single_raw = _yf_download_with_retries(tickers=[symbol], start=fetch_start, end=yf_end_ts)
			single_adj_close = _extract_adj_close(single_raw, symbols=[symbol])
			if single_adj_close.empty or symbol not in single_adj_close.columns:
				stooq_series = _download_from_stooq(symbol=symbol, start=fetch_start, end=end_ts)
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
				stooq_series = _download_from_stooq(symbol=symbol, start=fetch_start, end=end_ts)
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

	batch_part = pd.concat(batch_frames, axis=1) if batch_frames else pd.DataFrame()
	if not batch_part.empty:
		for symbol in batch_part.columns:
			_save_cached_symbol_prices(symbol=str(symbol), series=batch_part[str(symbol)].dropna())

	combined_parts: list[pd.DataFrame] = []
	if cached_frames:
		combined_parts.extend(cached_frames)
	if not batch_part.empty:
		combined_parts.append(batch_part)

	combined = pd.concat(combined_parts, axis=1) if combined_parts else pd.DataFrame()
	if fallback_frames:
		combined = pd.concat([combined, *fallback_frames], axis=1)
	combined = combined[~combined.index.duplicated(keep="last")]

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

