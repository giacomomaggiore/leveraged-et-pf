from __future__ import annotations

from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at runtime
    go = None


def _require_plotly() -> None:
    if go is None:
        raise ImportError(
            "plotly is required for visuals.py. Install it with 'pip install plotly'."
        )


def _validate_wealth_paths(wealth_paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(wealth_paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError("wealth_paths must have shape (paths, time+1).")
    if arr.shape[0] <= 0 or arr.shape[1] <= 1:
        raise ValueError("wealth_paths must include at least one path and two time points.")
    if not np.isfinite(arr).all():
        raise ValueError("wealth_paths contains NaN or inf values.")
    if np.any(arr <= 0.0):
        raise ValueError("wealth_paths must be strictly positive.")
    return arr


def _compute_drawdowns(wealth_paths: np.ndarray) -> np.ndarray:
    high_water_mark = np.maximum.accumulate(wealth_paths, axis=1)
    return (wealth_paths / high_water_mark) - 1.0


def plot_spaghetti_paths(
    wealth_paths: np.ndarray,
    n_sample: int = 100,
    *,
    seed: int | None = None,
    normalize_to_1: bool = False,
    title: str = "Monte Carlo Spaghetti Plot",
) -> Any:
    """Plot a random sample of equity curves.

    Parameters
    ----------
    wealth_paths:
        Array with shape (paths, time+1).
    n_sample:
        Number of paths to draw (typically 50-100).
    seed:
        Optional seed for reproducible path sampling.
    normalize_to_1:
        If True, each path is normalized by its first value.
    title:
        Figure title.
    """
    _require_plotly()
    wealth = _validate_wealth_paths(wealth_paths)

    if n_sample <= 0:
        raise ValueError("n_sample must be a positive integer.")

    n_paths, n_cols = wealth.shape
    n_pick = min(n_sample, n_paths)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_paths, size=n_pick, replace=False)

    y_matrix = wealth[chosen].copy()
    if normalize_to_1:
        y_matrix = y_matrix / y_matrix[:, [0]]

    x = np.arange(n_cols)
    fig = go.Figure()

    for idx in range(n_pick):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_matrix[idx],
                mode="lines",
                line={"width": 1},
                opacity=0.35,
                showlegend=False,
                hovertemplate="Day %{x}<br>Value %{y:,.2f}<extra></extra>",
            )
        )

    y_label = "Normalized Wealth" if normalize_to_1 else "Portfolio Value"
    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title=y_label,
        template="plotly_white",
    )
    return fig


def plot_terminal_wealth_distribution(
    wealth_paths: np.ndarray,
    bins: int = 50,
    *,
    title: str = "Distribution of Terminal Wealth",
) -> Any:
    """Plot histogram of terminal wealth across all Monte Carlo paths."""
    _require_plotly()
    wealth = _validate_wealth_paths(wealth_paths)

    if bins <= 0:
        raise ValueError("bins must be a positive integer.")

    terminal = wealth[:, -1]

    fig = go.Figure(
        data=[
            go.Histogram(
                x=terminal,
                nbinsx=bins,
                marker={"line": {"width": 0.5, "color": "white"}},
                opacity=0.85,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Terminal Portfolio Value",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def plot_drawdown_chart(
    wealth_paths: np.ndarray,
    drawdowns: np.ndarray | None = None,
    *,
    title: str = "Drawdown Chart (Median vs Worst)",
) -> Any:
    """Plot median and worst-case drawdown curves over time."""
    _require_plotly()
    wealth = _validate_wealth_paths(wealth_paths)

    if drawdowns is None:
        dd = _compute_drawdowns(wealth)
    else:
        dd = np.asarray(drawdowns, dtype=float)
        if dd.shape != wealth.shape:
            raise ValueError("drawdowns must have same shape as wealth_paths.")

    median_dd = np.median(dd, axis=0) * 100.0
    worst_dd = np.min(dd, axis=0) * 100.0
    x = np.arange(wealth.shape[1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median_dd,
            mode="lines",
            name="Median Drawdown",
            line={"width": 2},
            hovertemplate="Day %{x}<br>Median DD %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=worst_dd,
            mode="lines",
            name="Worst-Case Drawdown",
            line={"width": 2},
            hovertemplate="Day %{x}<br>Worst DD %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Trading Day",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
    )
    return fig
