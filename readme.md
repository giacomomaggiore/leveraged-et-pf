# Leveraged ETF Portfolio Simulator

This project builds a modular Python simulation framework to evaluate long-run behavior of daily leveraged ETFs (LETFs) in passive portfolios.

The simulator is designed around three critical realities:
- volatility drag from daily re-leveraging,
- dynamic borrowing/swap costs,
- tax path dependency via average cost basis tracking (PMC method).

Primary spec source: [instructions.md](instructions.md).

## Objective

Model and compare portfolio outcomes under realistic daily mechanics, then summarize distributional outcomes across Monte Carlo scenarios.

## Architecture

Planned modules:
- [data_loader.py](data_loader.py): price/rate ingestion, cleaning, and alignment.
- letf_engine.py: synthetic LETF daily return construction.
- montecarlo.py: scenario generation (parametric and bootstrap).
- portfolio_sim.py: iterative portfolio + rebalance + tax engine.
- metrics.py: pathwise risk/return statistics.
- visuals.py: charts for simulation diagnostics.
- [main.ipynb](main.ipynb): orchestration and exploration notebook.

## Key Financial Conventions (From Spec)

- Trading calendar: trading days only (no weekend/holiday synthetic rows).
- Rate conversion: annualized rates converted using 252 trading days.
- LETF cost inputs: decimal rates (example: 0.0095 = 0.95%).
- LETF costs applied only on trading days.
- Monte Carlo reproducibility: optional random seed.
- Monte Carlo output shape: `(paths, time, assets)`.
- Rebalance order (mandatory): sell -> pay taxes -> buy.
- Shares: fractional shares allowed.
- Costs/frictions: transaction costs and slippage are zero by default.
- Hedge overlay: none (prototype scope).

## LETF Return Model

Daily synthetic LETF return is defined as:

$$
R_{L,t} = L \cdot R_{i,t} - \left[ \frac{TER}{252} + (L-1) \cdot \frac{Rate_t + Spread}{252} \right]
$$

where:
- $R_{i,t}$: underlying asset daily return,
- $L$: leverage factor,
- $TER$: annual total expense ratio (decimal),
- $Rate_t$: annual borrowing/swap benchmark rate at time $t$ (decimal),
- $Spread$: annual spread (decimal).

## Tax Model (PMC / Average Cost Basis)

For each asset, track shares $Q$ and average cost basis $PMC$.

Buy update:

$$
PMC_{new} = \frac{PMC_{old} \cdot Q_{old} + P_{buy} \cdot Q_{buy}}{Q_{old} + Q_{buy}}
$$

Sell realized gain:

$$
Gain = (P_{sell} - PMC_{old}) \cdot Q_{sell}
$$

If $Gain > 0$, tax paid is $Gain \cdot TaxRate$ immediately during rebalance.

## Metrics Scope

Required pathwise metrics:
- CAGR,
- Maximum drawdown,
- Drawdown duration (recovery time),
- Sharpe ratio,
- Sortino ratio,
- Probability of ruin,
- Ulcer index.

Conventions:
- Use arithmetic daily returns for Sharpe/Sortino.
- Use geometric compounding for CAGR.
- Annualization uses 252 trading days.
- Drawdown recovery only when prior peak is reached or exceeded.

## Resources 
