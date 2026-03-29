from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _validate_historical_returns(historical_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitize historical return matrix."""
    if not isinstance(historical_returns, pd.DataFrame):
        raise TypeError("historical_returns must be a pandas DataFrame.")
    if historical_returns.empty:
        raise ValueError("historical_returns is empty.")

    clean = historical_returns.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        raise ValueError("historical_returns has no valid rows after cleaning.")
    return clean.astype(float)


def _robust_cholesky(cov: np.ndarray) -> np.ndarray:
    """Compute a stable Cholesky factor, adding tiny diagonal jitter if needed."""
    # Cholesky decomposition factors a covariance matrix as L @ L.T,
    # where L is lower triangular. If cov is not positive semi-definite,
    # this will raise a LinAlgError. We can add a small jitter to the diagonal
    # to try to fix near-singularity issues. We attempt this up to 6 times
    # with increasing jitter before giving up.
    
    
    # we Random draw from normal/t distributions
    # they are initially independent (no correlation).
    # we multiply them by  L so they have target correlatios

    
    eye = np.eye(cov.shape[0], dtype=float)
    jitter = 0.0
    for _ in range(6):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0

    raise np.linalg.LinAlgError("Covariance matrix is not positive semi-definite.")


def simulate_parametric_paths(
    historical_returns: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    distribution: Literal["normal", "student_t"] = "normal",
    student_t_df: float = 6.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate correlated synthetic returns using mean, covariance, and Cholesky."""
    clean = _validate_historical_returns(historical_returns)

    if n_paths <= 0 or horizon_days <= 0:
        raise ValueError("n_paths and horizon_days must be positive integers.")

    # checks for distribution input
    dist = distribution.lower()
    if dist not in {"normal", "student_t"}:
        raise ValueError("distribution must be 'normal' or 'student_t'.")
    if dist == "student_t" and student_t_df <= 2.0:
        raise ValueError("student_t_df must be > 2 to ensure finite variance.")

    # set up random number generator with optional seed for reproducibility
    rng = np.random.default_rng(seed)

    
    values = clean.to_numpy()
    n_assets = values.shape[1]

    # compute mean vector and covariance matrix from historical returns
    mu = values.mean(axis=0)
    cov = np.cov(values, rowvar=False, ddof=1)
    chol = _robust_cholesky(cov)

    if dist == "normal":
        # For normal distribution, we can directly use standard normal draws.
        innovations = rng.standard_normal(size=(n_paths, horizon_days, n_assets))
    else:
        # For student_t, we draw from the standard t distribution and scale to have the correct covariance.
        innovations = rng.standard_t(df=student_t_df, size=(n_paths, horizon_days, n_assets))
        innovations *= np.sqrt((student_t_df - 2.0) / student_t_df)

    # Apply the Cholesky factor to introduce correlations, and add the mean vector.
    correlated = innovations @ chol.T
    simulated = correlated + mu[None, None, :]
    return simulated


def simulate_bootstrap_paths(
    historical_returns: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    seed: int | None = None,
) -> np.ndarray:
    
    """Sample full historical daily rows with replacement (default method)."""
    clean = _validate_historical_returns(historical_returns)

    if n_paths <= 0 or horizon_days <= 0:
        raise ValueError("n_paths and horizon_days must be positive integers.")


    rng = np.random.default_rng(seed)
    values = clean.to_numpy()
    n_hist_days = values.shape[0]

    sampled_idx = rng.integers(0, n_hist_days, size=(n_paths, horizon_days))
    simulated = values[sampled_idx]
    return simulated


def simulate_monte_carlo(
    historical_returns: pd.DataFrame,
    n_paths: int,
    horizon_days: int,
    method: Literal["bootstrap", "parametric"] = "bootstrap",
    distribution: Literal["normal", "student_t"] = "normal",
    student_t_df: float = 6.0,
    seed: int | None = None,
) -> np.ndarray:
    """Main API returning simulated paths with shape (paths, time, assets)."""
    selected_method = method.lower()

    if selected_method == "bootstrap":
        return simulate_bootstrap_paths(
            historical_returns=historical_returns,
            n_paths=n_paths,
            horizon_days=horizon_days,
            seed=seed,
        )

    if selected_method == "parametric":
        return simulate_parametric_paths(
            historical_returns=historical_returns,
            n_paths=n_paths,
            horizon_days=horizon_days,
            distribution=distribution,
            student_t_df=student_t_df,
            seed=seed,
        )

    raise ValueError("method must be 'bootstrap' or 'parametric'.")