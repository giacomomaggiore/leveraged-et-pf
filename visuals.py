from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from metrics import _compute_drawdowns, _validate_wealth_paths

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at runtime
    go = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime
    plt = None


def plot_spaghetti_paths(
    wealth_paths: np.ndarray,
    n_sample: int = 100,
    *,
    seed: int | None = None,
    normalize_to_1: bool = False,
    title: str = "Monte Carlo Spaghetti Plot",
    subtitle: str | None = None,
    bottom_note: str | None = None,
    subtitle_align: Literal["left", "center", "right"] = "center",
    bottom_note_align: Literal["left", "center", "right"] = "left",
    bottom_note_x: float | None = None,
    bottom_note_y: float | None = None,
    bottom_note_box: bool = True,
    backend: Literal["plotly", "matplotlib"] = "plotly",
    width: int = 1650,
    height: int = 980,
    figsize: tuple[float, float] = (5.0, 5.5),
    dpi: int = 170,
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
    subtitle:
        Optional subtitle rendered below the title (supports HTML line breaks).
    bottom_note:
        Optional note rendered below the chart area (supports HTML line breaks).
    subtitle_align:
        Horizontal alignment for subtitle: "left", "center", or "right".
    bottom_note_align:
        Horizontal alignment for bottom note: "left", "center", or "right".
    bottom_note_x:
        Optional x position in paper coordinates for bottom note text box.
    bottom_note_y:
        Optional y position in paper coordinates for bottom note text box.
    bottom_note_box:
        If True, render the bottom note inside a bordered text box.
    """
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

    if backend == "matplotlib":
        if plt is None:
            raise ImportError(
                "matplotlib is required for matplotlib visuals. Install it with 'pip install matplotlib'."
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for idx in range(n_pick):
            ax.plot(
                x,
                y_matrix[idx],
                lw=1.0,
                alpha=0.25,
                color="#1f4e79",
            )

        y_label = "Normalized Wealth" if normalize_to_1 else "Portfolio Value"
        ax.set_xlabel("Trading Day")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.set_facecolor("#fbfcfe")
        fig.patch.set_facecolor("#f6f8fb")
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.97)

        if subtitle:
            fig.text(0.5, 0.93, subtitle.replace("<br>", "\n"), ha="center", va="top", fontsize=11, color="#334155")
        if bottom_note:
            align_map = {"left": "left", "center": "center", "right": "right"}
            x_map = {"left": 0.01, "center": 0.5, "right": 0.99}
            note_x = x_map[bottom_note_align] if bottom_note_x is None else float(bottom_note_x)
            note_y = 0.02 if bottom_note_y is None else float(bottom_note_y)
            fig.text(
                note_x,
                note_y,
                bottom_note.replace("<br>", "\n"),
                ha=align_map[bottom_note_align],
                va="bottom",
                fontsize=10,
                color="#334155",
            )

        fig.subplots_adjust(top=0.86, bottom=0.22, left=0.08, right=0.98)
        return fig

    if go is None:
        raise ImportError(
            "plotly is required for visuals.py. Install it with 'pip install plotly'."
        )
    fig = go.Figure()

    for idx in range(n_pick):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_matrix[idx],
                mode="lines",
                line={"width": 1.1},
                opacity=0.55,
                showlegend=False,
                hovertemplate="Day %{x}<br>Value %{y:,.2f}<extra></extra>",
            )
        )

    valid_alignments = {"left", "center", "right"}
    if subtitle_align not in valid_alignments:
        raise ValueError("subtitle_align must be one of: left, center, right.")
    if bottom_note_align not in valid_alignments:
        raise ValueError("bottom_note_align must be one of: left, center, right.")

    x_positions = {"left": 0.0, "center": 0.5, "right": 1.0}
    x_anchors = {"left": "left", "center": "center", "right": "right"}

    y_label = "Normalized Wealth" if normalize_to_1 else "Portfolio Value"

    layout_kwargs: dict[str, Any] = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis_title": "Trading Day",
        "yaxis_title": y_label,
        "template": "plotly_white",
        "width": width,
        "height": height,
        "font": {"family": "Avenir Next, Helvetica, Arial, sans-serif", "size": 14, "color": "#1f2937"},
        "plot_bgcolor": "#f9fbfd",
        "paper_bgcolor": "#f4f7fb",
        "xaxis": {"gridcolor": "#dbe3ee", "zeroline": False},
        "yaxis": {"gridcolor": "#dbe3ee", "zeroline": False},
    }

    if subtitle:
        layout_kwargs["margin"] = {"t": 200, "l": 80, "r": 40}
        fig.add_annotation(
            x=x_positions[subtitle_align],
            y=1.06,
            xref="paper",
            yref="paper",
            text=subtitle,
            showarrow=False,
            align=subtitle_align,
            xanchor=x_anchors[subtitle_align],
            yanchor="bottom",
            font={"size": 14, "color": "#374151"},
        )

    if bottom_note:
        existing_margin = dict(layout_kwargs.get("margin", {}))
        existing_margin["b"] = 180
        layout_kwargs["margin"] = existing_margin
        note_x = x_positions[bottom_note_align] if bottom_note_x is None else float(bottom_note_x)
        note_y = -0.25 if bottom_note_y is None else float(bottom_note_y)

        annotation_kwargs: dict[str, Any] = {
            "x": note_x,
            "y": note_y,
            "xref": "paper",
            "yref": "paper",
            "text": bottom_note,
            "showarrow": False,
            "align": bottom_note_align,
            "xanchor": x_anchors[bottom_note_align],
            "yanchor": "top",
            "font": {"size": 11, "color": "#0f172a"},
        }
        if bottom_note_box:
            annotation_kwargs.update(
                {
                    "bgcolor": "rgba(255, 255, 255, 0.82)",
                    "bordercolor": "#cbd5e1",
                    "borderwidth": 1,
                }
            )
        fig.add_annotation(
            **annotation_kwargs,
        )

    fig.update_layout(**layout_kwargs)
    return fig


def plot_terminal_wealth_distribution(
    wealth_paths: np.ndarray,
    bins: int = 50,
    *,
    title: str = "Distribution of Terminal Wealth",
    backend: Literal["plotly", "matplotlib"] = "plotly",
    width: int = 1350,
    height: int = 760,
    figsize: tuple[float, float] = (4.0, 8.0),
    dpi: int = 170,
) -> Any:
    """Plot histogram of terminal wealth across all Monte Carlo paths."""
    wealth = _validate_wealth_paths(wealth_paths)

    if bins <= 0:
        raise ValueError("bins must be a positive integer.")

    terminal = wealth[:, -1]

    if backend == "matplotlib":
        if plt is None:
            raise ImportError(
                "matplotlib is required for matplotlib visuals. Install it with 'pip install matplotlib'."
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.hist(terminal, bins=bins, color="#2b6cb0", edgecolor="white", linewidth=0.7, alpha=0.9)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
        ax.set_xlabel("Terminal Portfolio Value")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_facecolor("#fbfcfe")
        fig.patch.set_facecolor("#f6f8fb")
        fig.tight_layout()
        return fig

    if go is None:
        raise ImportError(
            "plotly is required for visuals.py. Install it with 'pip install plotly'."
        )

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
        width=width,
        height=height,
        font={"family": "Avenir Next, Helvetica, Arial, sans-serif", "size": 14, "color": "#1f2937"},
        plot_bgcolor="#f9fbfd",
        paper_bgcolor="#f4f7fb",
    )
    return fig


def plot_drawdown_chart(
    wealth_paths: np.ndarray,
    drawdowns: np.ndarray | None = None,
    *,
    title: str = "Drawdown Chart (Median vs Worst)",
    backend: Literal["plotly", "matplotlib"] = "plotly",
    width: int = 1350,
    height: int = 760,
    figsize: tuple[float, float] = (14.0, 8.0),
    dpi: int = 170,
) -> Any:
    """Plot median and worst-case drawdown curves over time."""
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

    if backend == "matplotlib":
        if plt is None:
            raise ImportError(
                "matplotlib is required for matplotlib visuals. Install it with 'pip install matplotlib'."
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(x, median_dd, lw=2.2, color="#1f77b4", label="Median Drawdown")
        ax.plot(x, worst_dd, lw=2.2, color="#d62728", label="Worst-Case Drawdown")
        ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(loc="best", frameon=False)
        ax.set_facecolor("#fbfcfe")
        fig.patch.set_facecolor("#f6f8fb")
        fig.tight_layout()
        return fig

    if go is None:
        raise ImportError(
            "plotly is required for visuals.py. Install it with 'pip install plotly'."
        )

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
        width=width,
        height=height,
        font={"family": "Avenir Next, Helvetica, Arial, sans-serif", "size": 14, "color": "#1f2937"},
        plot_bgcolor="#f9fbfd",
        paper_bgcolor="#f4f7fb",
    )
    return fig


def save_matplotlib_figure(fig: Any, output_path: str | Path, *, dpi: int = 220) -> None:
    """Save a matplotlib figure with a tight bounding box for long notes/subtitles."""
    if plt is None:
        raise ImportError(
            "matplotlib is required for matplotlib visuals. Install it with 'pip install matplotlib'."
        )
    fig.savefig(Path(output_path), dpi=dpi, bbox_inches="tight")
