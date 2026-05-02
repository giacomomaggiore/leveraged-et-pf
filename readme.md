# Leveraged ETF Portfolio Simulator

A comprehensive, production-grade Python simulation framework for evaluating the long-term viability of daily leveraged ETFs (LETFs) in passive portfolios. This system implements a fully realistic **day-by-day** simulation engine that captures volatility drag, dynamic borrowing costs, exact capital gains taxation, and path dependency across Monte Carlo scenarios.

## Executive Summary

The simulator addresses three critical financial realities that most portfolio analyses overlook:

1. **Volatility Drag from Daily Re-leveraging**: LETFs reset their leverage ratio daily, causing performance slippage on volatile days. This compounding effect erodes returns over time.
2. **Dynamic Borrowing/Swap Costs**: Financing costs vary daily based on market conditions (FRED rates, swap spreads), requiring precise daily tracking rather than flat assumptions.
3. **Tax Path Dependency via PMC Tracking**: The project implements exact capital gains taxation using the Prezzo Medio di Carico (PMC / Average Cost Basis) method, tracking cost basis on a per-asset basis during rebalancing.

## Project Objectives

- **Accurate LETF Modeling**: Synthetic daily LETF returns computed from underlying returns, leverage factor, annual TER, and daily financing costs.
- **Flexible Scenario Generation**: Support both parametric (normal/t-distribution) and historical bootstrap Monte Carlo methods.
- **Exact Portfolio Mechanics**: Day-by-day simulation with annual rebalancing, tolerance bands, and fractional shares.
- **Comprehensive Risk Metrics**: Compute CAGR, max drawdown, recovery time, Sharpe/Sortino ratios, probability of ruin, and Ulcer Index across pathwise distributions.
- **Configurable Portfolio Construction**: Accept arbitrary spot ETFs and synthetic LETFs with target allocations and rebalancing rules.

## Architecture Overview

The project follows a **modular, functional design** with five core computation layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         main.ipynb / main.py                           │
│                  (Orchestration & Visualization)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                      orchestration.py                                   │
│           (Configuration Management & Simulation Workflow)             │
├─────────────────────────────────────────────────────────────────────────┤
│  data_loader.py    │  montecarlo.py   │  letf_engine.py  │ metrics.py  │
│  (Data Ingestion)  │  (Path Generation)│  (LETF Pricing)  │  (Analysis) │
├─────────────────────────────────────────────────────────────────────────┤
│                      portfolio_sim.py                                   │
│        (Core Iterative Engine: Rebalancing, Tax, PMC Tracking)         │
├─────────────────────────────────────────────────────────────────────────┤
│                  visuals.py (Plotting & Diagnostics)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose | Key Responsibilities |
|--------|---------|----------------------|
| **data_loader.py** | Market data ingestion | Download prices from yfinance/defeatbeta; fetch FRED rates; align all series on trading-day calendar; convert annual rates to daily |
| **letf_engine.py** | LETF synthetic return construction | Apply leverage, TER, and dynamic borrowing costs to underlying returns using the specification formula |
| **montecarlo.py** | Scenario path generation | Generate correlated multi-asset return paths using Cholesky-based parametric or historical bootstrap methods |
| **portfolio_sim.py** | Core simulation engine | Day-by-day portfolio evolution; annual rebalancing with tolerance bands; PMC-based capital gains tracking; tax payment sequencing |
| **metrics.py** | Performance analytics | Compute pathwise CAGR, volatility, drawdowns, Sharpe/Sortino ratios, probability of ruin, and aggregate distribution statistics |
| **orchestration.py** | Simulation orchestration | Manage simulation workflows, configuration validation, asset/rate alignment, and results aggregation |
| **visuals.py** | Diagnostics visualization | Spaghetti plots, terminal wealth distributions, drawdown curves |
| **run_portfolio_batch.py** | Batch scenario runner | Execute multiple portfolio configurations with local data caching; aggregate results to CSV |
| **constants.py** | Shared constants | Trading days per year (252) |

## Key Financial Conventions

### Trading Calendar & Rate Conventions
- **Trading Calendar**: The simulator operates on trading-days-only (no weekend/holiday synthetic rows). All datetime indices are `pd.DatetimeIndex` with trading-day granularity.
- **Rate Conversion**: Annualized rates are converted to daily using **252 trading days per year**. For any annualized rate $r_{ann}$, the daily equivalent is $r_{daily} = r_{ann} / 252$.
- **Cost Units**: All cost inputs (TER, borrowing rates, spreads) are specified in **decimal form** (e.g., 0.0095 = 95 basis points = 0.95%).
- **Application**: LETF costs, taxes, and rebalancing transactions occur only on trading days.

### Rebalancing Policy
- **Frequency**: Rebalancing is triggered **annually** (default every 252 trading days).
- **Trigger Condition**: If *any* asset's current weight deviates from its target weight by more than the `tolerance_band` (default 5%), the entire portfolio is rebalanced.
- **Execution**: Rebalancing follows a strict **three-step sequence**: (1) Sell, (2) Pay Taxes, (3) Buy. This ensures accurate PMC tracking and tax liability calculation.

### Shares & Frictions
- **Fractional Shares**: The simulator supports fractional shares (no rounding), enabling precise portfolio construction.
- **Transactional Frictions**: Transaction costs, slippage, and bid-ask spreads are assumed **zero by default** (can be extended).
- **Hedge Overlays**: No derivative hedges are used in this prototype scope.

---

## Detailed Module Specifications

### 1. Data Loader (`data_loader.py`)

**Purpose**: Ingest, clean, and align market data from multiple sources on a trading-day calendar.

**Data Sources**:
- **Equities/Bonds**: Downloaded via `yfinance` (adjusted close prices).
- **Risk-Free / Borrowing Rates**: Downloaded from FRED via the official FRED API (e.g., EFFR for daily effective fed funds rate).
- **Fallback Provider**: `defeatbeta_api` provides fallback data when yfinance is rate-limited or unavailable.

**Data Processing Pipeline**:

1. **Cache Layer**: Prices are cached locally in `data/` directory. If a symbol's cache exists and covers the requested date range (or overlaps), cached data is loaded first.
2. **Online Fetch**: Missing or stale data is fetched online with exponential backoff and jitter to handle rate limits gracefully.
3. **Daily Return Calculation**: Simple daily returns are computed as $r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$.
4. **Trading-Day Alignment**: All price series and rate series are reindexed to a common trading-day calendar. Forward-fill is applied sparingly (only on trading-day gaps after merge).
5. **Rate Conversion**: Annualized rates (e.g., EFFR in % form) are converted to decimal daily rates: $r_{daily} = \frac{r_{annual}}{100 \times 252}$.

**Output**: A cleaned pandas DataFrame of daily returns (shape: `(days, assets)`) and a series of daily risk-free rates (shape: `(days,)`).

### 2. LETF Engine (`letf_engine.py`)

**Purpose**: Compute synthetic daily returns for leveraged ETFs accounting for volatility drag and financing costs.

**The LETF Return Formula**:

The daily return of a synthetic LETF with leverage factor $L$ is computed as:

$$R_{L,t} = L \cdot R_{i,t} - \left[ \frac{TER}{252} + (L-1) \cdot \frac{Rate_t + Spread}{252} \right]$$

Where:
- $R_{i,t}$: Underlying asset's daily simple return.
- $L$: Leverage factor (e.g., 2.0 for 2x LETF, 3.0 for 3x LETF).
- $TER$: Annual Total Expense Ratio (decimal form, e.g., 0.0092 = 92 basis points).
- $Rate_t$: Daily borrowing/swap benchmark rate at time $t$ (already converted to daily decimal).
- $Spread$: Annual spread in decimal form; represents the difference between the benchmark and actual borrowing cost.
- Daily cost term: $\frac{TER + (L-1) \cdot (Rate_t + Spread)}{252}$.

**Financial Intuition**:
- The **first term** ($L \cdot R_{i,t}$) is the "natural" leverage effect: daily returns are amplified by the leverage factor.
- The **second term** (financing costs) represents the drag from daily costs:
  - $\frac{TER}{252}$: Management costs, whether the portfolio is leveraged or not.
  - $(L-1) \cdot \frac{Rate_t + Spread}{252}$: Borrowing costs scale with the *incremental* leverage $(L-1)$. A 1x (unleveraged) asset has no borrowing cost; 2x has $(2-1)$ times the cost; 3x has $(3-1)$ times the cost.

**Volatility Drag Effect**:
While the formula appears simple, the *compounding effect* of daily rebalancing combined with volatility creates "drag." On a high-volatility day where the underlying falls 5%, a 2x LETF falls ~10% (minus financing costs), but must then rebalance to maintain exactly 2x leverage. This rebalancing at lower prices locks in losses, a phenomenon known as **path-dependent decay** or "beta slippage."

**Implementation Details**:
- Input: A pandas Series of underlying daily returns (trading-day indexed).
- Parameters: `leverage`, `ter`, `borrowing_rate` (as Series or scalar), `spread`.
- Output: A pandas Series of synthetic LETF daily returns with the same index.
- Validation: Ensures all inputs are finite, leverage > 0, and borrowing rates align to the underlying return calendar.

### 3. Monte Carlo Engine (`montecarlo.py`)

**Purpose**: Generate correlated multi-asset return paths for stochastic simulation.

#### Method A: Parametric (Synthetic Distribution)

**Process**:

1. **Historical Calibration**: Compute mean returns $\mu$ and covariance matrix $\Sigma$ from historical daily returns.
2. **Cholesky Decomposition**: Factorize $\Sigma = L L^T$ where $L$ is lower-triangular. Add small diagonal jitter (up to 1e-10) if $\Sigma$ is nearly singular.
3. **Innovation Generation**: 
   - For normal distribution: Draw independent standard normals $Z \sim \mathcal{N}(0,1)$ with shape `(paths, horizon, assets)`.
   - For t-distribution: Draw independent Student-t variates $T$ with df degrees of freedom, scaled to match variance: $T \cdot \sqrt{\frac{df-2}{df}}$.
4. **Correlation Introduction**: Compute $Y = Z \cdot L^T$ to introduce historical correlations.
5. **Drift Addition**: Add historical mean: $R = Y + \mu$.

**Mathematical Detail**:
If $Z$ is `(N, T, A)` standard normals (paths $N$, time steps $T$, assets $A$):
$$R_{n,t,a} = \sum_{b=1}^{A} Z_{n,t,b} \cdot L_{a,b} + \mu_a$$

This preserves both the historical mean and covariance while generating synthetic paths.

**Advantages**: 
- Fast generation for large numbers of paths.
- Can switch distributions to test sensitivity to tail risk.

**Limitations**: 
- Assumes that historical correlations and mean/covariance are stationary.
- May underestimate realized kurtosis and left-tail risk compared to actual markets.

#### Method B: Historical Bootstrap (DEFAULT)

**Process**:

1. **Resample Rows**: Randomly sample entire rows (days) from the historical return dataframe **with replacement**.
2. **Preserve Correlation**: By resampling full rows, the exact historical correlation matrix and joint distribution (including skewness, kurtosis, and tail dependence) are automatically preserved.
3. **No Parametric Assumption**: The method is non-parametric; it does not assume normality or any particular distribution.

**Mathematical Detail**:
For each MC path $n$ and time step $t$, randomly select a historical row index $i_t \in \{1, 2, \ldots, T_{hist}\}$ uniformly. The path's returns at step $t$ are then $R_{n,t} = R_{i_t}$ (a full row from history).

**Advantages**:
- Captures fat tails, skewness, and real-world correlations organically.
- No parametric assumptions; purely data-driven.
- Provides natural "regime switching" if historical data includes market shocks.

**Limitations**:
- Limited to the historical sample size; cannot extrapolate beyond observed patterns.
- Slower than parametric for very large path counts (though still fast on modern hardware).

**Output**: A 3D numpy array of shape `(paths, horizon_days, assets)` containing daily returns for each simulated path.

### 4. Portfolio Simulator (`portfolio_sim.py`)

**Purpose**: Core iterative engine that simulates day-by-day portfolio evolution with rebalancing and exact tax tracking.

**Conceptual Flow**:

For each Monte Carlo path $p = 1, \ldots, N$:
1. Initialize: Set shares $Q = \frac{initial\_capital \times w_{target}}{price}$; set PMC $= price$ for each asset.
2. For each trading day $t = 1, \ldots, T$:
   - **Update Prices**: $Price_t = Price_{t-1} \times (1 + R_t)$ where $R_t$ is the simulated return.
   - **Calculate Wealth**: $Wealth_t = \sum_i Q_i \times Price_{i,t} + Cash$.
   - **Check Rebalance Trigger**: If day $t$ is a rebalance date AND any weight drifts by > tolerance_band:
     - Execute rebalance (see below).

**Rebalancing Process** (Three-Step Sequence):

**Step 1: SELL**
- Calculate target values: $V_{target,i} = Wealth \times w_{target,i}$.
- For each asset where current value $< target$ value, sell the shortfall:
  - Sell quantity: $Q_{sell} = \frac{|V_{target,i} - V_{current,i}|}{Price_i}$.
  - Update shares: $Q_i \leftarrow Q_i - Q_{sell}$.
  - Accumulate cash: $Cash \leftarrow Cash + Q_{sell} \times Price_i$.

**Step 2: PAY TAXES**
- For each sold asset, compute realized gain:
  $$Gain = (Price_{sell} - PMC_{old}) \times Q_{sell}$$
- Accumulate positive gains: $TaxableGain = \sum (\max(Gain, 0))$.
- Compute tax due: $Tax = TaxableGain \times TaxRate$.
- Deduct from cash: $Cash \leftarrow Cash - Tax$.

**Step 3: BUY**
- For each asset where target value $> current$ value, allocate cash:
  - Target buy value: $V_{buy,i} = \max(V_{target,i} - V_{current,i}, 0)$.
  - Buy quantity: $Q_{buy} = \frac{V_{buy,i}}{Price_i}$.
  - Update shares: $Q_i \leftarrow Q_i + Q_{buy}$.
  - Update PMC (average cost basis):
    $$PMC_{new} = \frac{PMC_{old} \times Q_{old} + Price_i \times Q_{buy}}{Q_{old} + Q_{buy}}$$
  - Deduct from cash: $Cash \leftarrow Cash - V_{buy,i}$.

**Tax Tracking (PMC Method)**:

The Prezzo Medio di Carico (PMC / weighted average cost basis) method is used:

**On Purchase**:
$$PMC_{new} = \frac{PMC_{old} \cdot Q_{old} + Price_{buy} \cdot Q_{buy}}{Q_{old} + Q_{buy}}$$

This blends the old and new cost bases proportionally.

**On Sale**:
$$Gain = (Price_{sell} - PMC_{old}) \times Q_{sell}$$

The cost basis used is the current PMC; after the sale, PMC remains unchanged (only quantities change).

**Why This Matters**:
- Exact tax tracking enables accurate after-tax portfolio valuation.
- PMC ensures that cost basis accurately reflects the weighted average price paid, not first-in-first-out (FIFO) or other conventions.
- Path dependency: Each path has unique rebalancing events, price movements, and tax consequences.

**Output**: A `SimulationResult` object containing:
- `wealth_paths`: Portfolio value at each time step (shape: `(paths, time+1)`).
- `portfolio_returns`: Day-by-day returns (shape: `(paths, time)`).
- `taxes_paid`: Cumulative tax paid per path per day (shape: `(paths, time)`).
- `rebalance_flags`: Boolean array indicating rebalance events (shape: `(paths, time)`).

### 5. Metrics Computation (`metrics.py`)

**Purpose**: Calculate comprehensive risk and return statistics from simulated wealth paths.

**Pathwise Metrics** (computed for each Monte Carlo path independently):

#### CAGR (Compound Annual Growth Rate)
$$CAGR = \left( \frac{Wealth_{end}}{Wealth_{start}} \right)^{\frac{252}{n\_days}} - 1$$

Where $n\_days$ is the number of days simulated. Annualization uses 252 trading days.

#### Volatility (Annualized)
Daily volatility is computed from daily portfolio returns:
$$\sigma_{daily} = \sqrt{\frac{1}{n\_days - 1} \sum_{t=1}^{n\_days} (r_{p,t} - \bar{r}_p)^2}$$

Annualized volatility: $\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$.

#### Maximum Drawdown
Drawdown at time $t$ is defined as:
$$DD_t = \frac{Wealth_t - HWM_t}{HWM_t}$$

Where $HWM_t$ (High-Water Mark) is the running maximum wealth: $HWM_t = \max(Wealth_0, Wealth_1, \ldots, Wealth_t)$.

Maximum Drawdown is the worst (most negative) drawdown: $MD = \min(DD_t)$.

**Interpretation**: A max drawdown of -40% means the portfolio fell to 60% of its peak value at worst.

#### Drawdown Duration (Recovery Time)
The number of consecutive trading days during which the portfolio is below its high-water mark (i.e., in drawdown). Recovery occurs when the portfolio regains its previous peak (or exceeds it).

**Calculation**: Count the longest run of days where $DD_t < 0$.

#### Sharpe Ratio
$$Sharpe = \frac{\mu(r_p - r_f)}{\sigma(r_p - r_f)} \times \sqrt{252}$$

Where:
- $r_p$: Daily portfolio returns.
- $r_f$: Daily risk-free rate (constant or time-varying).
- $\mu(\cdot)$: Mean.
- $\sigma(\cdot)$: Standard deviation (using unbiased sample variance, $ddof=1$).

**Interpretation**: Excess return per unit of excess volatility, annualized.

#### Sortino Ratio
$$Sortino = \frac{\mu(r_p - r_f)}{\sigma_{down}(r_p - r_f)} \times \sqrt{252}$$

Where $\sigma_{down}$ is the *downside standard deviation*, computed only from negative excess returns:
$$\sigma_{down} = \sqrt{\frac{1}{n\_down - 1} \sum_{t: r_{p,t} - r_f < 0} (r_{p,t} - r_f)^2}$$

**Why Sortino Over Sharpe**: Sortino penalizes only downside volatility (losses), not upside volatility. It's often preferred for skewed or tail-risk strategies.

#### Ulcer Index
A risk metric that emphasizes the depth and duration of drawdowns:
$$UI = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (\max(DD_t \times 100, 0))^2}$$

Where drawdowns are expressed as percentages. The Ulcer Index is particularly sensitive to prolonged underwater periods.

#### Probability of Ruin
The fraction of Monte Carlo paths where the portfolio falls below a ruin threshold (default: 10% of initial capital):
$$P(Ruin) = \frac{\#\{paths : Wealth_{min} \leq 0.10 \times Wealth_0\}}{N\_paths}$$

**Aggregation**: The module computes summary statistics across all paths:
- **Mean**: Average metric value across paths.
- **Median**: 50th percentile.
- **P5**: 5th percentile (worst 5% of scenarios).
- **P95**: 95th percentile (best 5% of scenarios).

**Output**: A `MetricsResult` object containing:
- `pathwise`: DataFrame with metrics for each path (shape: `(paths, metrics)`).
- `summary`: DataFrame with aggregated statistics (mean, median, p5, p95).
- `drawdowns`: Full drawdown matrix (shape: `(paths, time+1)`).

### 6. Orchestration (`orchestration.py`)

**Purpose**: Manage the full simulation workflow, from configuration validation through result aggregation.

**Configuration Hierarchy**:

```python
MarketDataConfig
  └─ start, end, fred_series, fred_is_percent

SimulationConfig
  ├─ market: MarketDataConfig
  ├─ assets: [SpotAssetConfig | SyntheticLETFAssetConfig]
  ├─ portfolio: PortfolioConfig
  │   ├─ target_weights: {asset_id: weight}
  │   ├─ initial_capital
  │   ├─ rebalance_frequency_days
  │   ├─ tolerance_band
  │   └─ capital_gains_tax_rate
  └─ monte_carlo: MonteCarloConfig
      ├─ n_paths, horizon_days
      ├─ method: "bootstrap" | "parametric"
      ├─ distribution: "normal" | "student_t"
      └─ seed
```

**Key Validations**:
- All asset IDs in `target_weights` must be defined in the `assets` list.
- Target weights must sum to a positive value (normalized internally).
- Asset IDs must be unique.
- Market data date range must accommodate the simulation horizon.

**Workflow**:

1. **Load Historical Data**: Fetch price history and risk-free rates; compute daily returns.
2. **Construct LETF Returns**: For each synthetic LETF asset, apply the LETF engine formula.
3. **Generate MC Paths**: Simulate correlated return paths using the specified method.
4. **Simulate Portfolio**: Apply the portfolio engine day-by-day.
5. **Compute Metrics**: Aggregate statistics across paths.
6. **Save Results**: Optionally write metrics to CSV.

### 7. Visualization (`visuals.py`)

**Plots**:

1. **Spaghetti Plot**: Displays 50–100 randomly selected equity curves from the MC simulation. Useful for eyeballing the distribution of outcomes.
2. **Terminal Wealth Distribution**: Histogram of final portfolio values across all paths. Shows the spread and skewness of long-term outcomes.
3. **Drawdown Curves**: Median and percentile (p5, p95) drawdown paths over time.

**Backends**: Both Plotly (interactive, web-based) and Matplotlib (static, traditional) are supported.

### 8. Batch Runner (`run_portfolio_batch.py`)

**Purpose**: Execute multiple portfolio configurations, leveraging local data caching to avoid redundant downloads.

**Features**:
- Infers the largest available local data window (common trading-day range across all symbols).
- Runs all portfolio variants on that window without re-downloading.
- Aggregates metrics to a single CSV file for easy comparison.
- Supports custom TER and spread defaults.

---

## Detailed Execution Workflow

### From Configuration to Results

1. **User defines a `SimulationConfig`** in a Jupyter notebook or Python script:
   - Specifies market data date range, FRED series, asset tickers, leverage factors, target weights, rebalancing rules, tax rate, MC parameters.

2. **Orchestration Module** (`orchestration.py`):
   - Validates configuration.
   - Calls `data_loader.load_market_data()` to fetch and align historical data.
   - Extracts daily returns and risk-free rates.

3. **LETF Engine** (`letf_engine.py`):
   - For each synthetic LETF, applies the formula: $R_{L,t} = L \cdot R_i - [\frac{TER}{252} + (L-1) \frac{Rate + Spread}{252}]$.
   - Returns a combined DataFrame of all asset returns (spot + synthetic LETF).

4. **Monte Carlo** (`montecarlo.py`):
   - Generates $N$ correlated return paths of length $T$ using the specified method (bootstrap or parametric).
   - Output shape: `(N, T, A)` where $A$ is the number of assets.

5. **Portfolio Simulator** (`portfolio_sim.py`):
   - Iterates through each path and each trading day.
   - Updates portfolio wealth, checks rebalancing triggers, executes buy/sell/tax sequence.
   - Tracks PMC for each asset to compute exact capital gains.

6. **Metrics Engine** (`metrics.py`):
   - Computes 8 pathwise metrics (CAGR, volatility, drawdown, etc.) for each path.
   - Aggregates to distribution statistics (mean, median, p5, p95).

7. **Results Aggregation**:
   - Optionally save metrics to CSV.
   - Optionally generate visualizations (spaghetti plots, wealth distributions).

### Example Usage (Pseudocode)

```python
from orchestration import SimulationConfig, MarketDataConfig, PortfolioConfig, MonteCarloConfig
from orchestration import SpotAssetConfig, SyntheticLETFAssetConfig, evaluate_portfolio_from_config

# Define the simulation
config = SimulationConfig(
    market=MarketDataConfig(start="2015-01-01", end="2023-12-31", fred_series="EFFR"),
    assets=[
        SpotAssetConfig(id="VTI", ticker="VTI"),
        SpotAssetConfig(id="BND", ticker="BND"),
        SyntheticLETFAssetConfig(id="VTI_2x", underlying_ticker="VTI", leverage=2.0, ter=0.0092, spread=0.003),
    ],
    portfolio=PortfolioConfig(
        target_weights={"VTI": 0.6, "BND": 0.2, "VTI_2x": 0.2},
        initial_capital=100_000,
        rebalance_frequency_days=252,
        tolerance_band=0.05,
        capital_gains_tax_rate=0.20,
    ),
    monte_carlo=MonteCarloConfig(n_paths=1000, horizon_days=252*10, method="bootstrap", seed=42),
)

# Run simulation
result = evaluate_portfolio_from_config(config)

# Access results
print(result.metrics.summary)  # Mean, median, p5, p95 for each metric
plot_spaghetti_paths(result.portfolio.wealth_paths, n_sample=100)
```

---

## Key Financial Insights

### Volatility Drag

LETFs do not perfectly replicate $L \times$ benchmark returns over time. The discrepancy arises from **daily rebalancing in volatile markets**:

- On days with high volatility, the LETF amplifies losses more than gains (asymmetry).
- Rebalancing at lower prices locks in the loss proportionally more than recovery at higher prices gains back proportionally.
- Over decades, this compounds into significant **beta slippage**.

**Example**: A 2x LETF tracking the S&P 500 in a 20% annual volatility environment loses roughly 1–2% per year to this effect (varies with correlation structure).

### Borrowing Cost Dynamics

Financing costs are *not* static. They vary with:
- **Benchmark Rate**: EFFR, Fed Funds, Libor, or swap rates change daily.
- **Spread**: The difference between the benchmark and what the fund actually pays can widen during market stress.
- **Leverage Factor**: Higher leverage multiplies the cost impact.

The simulator captures this path-dependency: a scenario with a Fed rate spike mid-simulation will see different outcomes than one without, all else equal.

### Tax Impact via PMC

Capital gains taxes reduce long-term returns significantly:

- **Buy-and-Hold**: Low tax because gains are realized only at the end.
- **Annual Rebalancing**: Every rebalance can trigger capital gains taxes if assets appreciated. The tax drag compounds.
- **High Volatility + High Leverage**: Frequent rebalancing and larger gains amplify tax drag.

The simulator's PMC tracking ensures that taxes are computed correctly: only profits above the weighted-average purchase price are taxed.

### Path Dependency & Monte Carlo

Because the portfolio simulator is **deterministic given a return path**, different return sequences lead to different outcomes *even with the same summary statistics*. This is path dependency:

- **Return Sequence Risk**: A portfolio that gains 20% and then loses 10% is NOT the same as one that loses 10% then gains 20%. (The first ends at ~$10.8k, the second at ~$10.8k if starting at $10k, but rebalancing behavior differs.)
- **Rebalancing Timing**: Rebalances occur on fixed dates, not when assets become misaligned. A crash on Dec 31 triggers a rebalance; a crash on Jan 1 doesn't (until the next annual rebalance).

Monte Carlo simulation captures this variability across thousands of paths, providing a distribution of outcomes rather than a single "expected" value.

---

## Key Formulas Reference

| Concept | Formula | Notes |
|---------|---------|-------|
| Daily LETF Return | $R_{L,t} = L \cdot R_i - [\frac{TER}{252} + (L-1) \frac{Rate + Spread}{252}]$ | Core LETF mechanics |
| PMC (Buy Update) | $PMC_{new} = \frac{PMC_{old} \cdot Q_{old} + P \cdot Q_{buy}}{Q_{old} + Q_{buy}}$ | Average cost basis tracking |
| Realized Gain (Sell) | $Gain = (P_{sell} - PMC_{old}) \times Q_{sell}$ | Taxable gain per share × quantity |
| CAGR | $CAGR = (\frac{Wealth_{end}}{Wealth_0})^{\frac{252}{n\_days}} - 1$ | Annualized growth rate |
| Volatility (Annual) | $\sigma = \sigma_{daily} \times \sqrt{252}$ | Daily vol annualized to trading year |
| Max Drawdown | $MD = \min_t (\frac{Wealth_t - HWM_t}{HWM_t})$ | Worst peak-to-trough decline |
| Sharpe Ratio | $Sharpe = \frac{\mu(r_p - r_f)}{\sigma(r_p - r_f)} \times \sqrt{252}$ | Excess return per unit of risk |
| Sortino Ratio | $Sortino = \frac{\mu(r_p - r_f)}{\sigma_{down}} \times \sqrt{252}$ | Penalizes downside vol only |

---

## Environment Setup

### Prerequisites
- Python 3.9+
- Dependencies listed in `requirements.txt`

### Installation

1. **Clone or download the repository:**
   ```bash
   cd /path/to/leveraged-etf-pf
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

- **yfinance**: Fetch historical price data from Yahoo Finance.
- **pandas**: Data manipulation and alignment.
- **numpy**: Numerical computation and vectorization.
- **plotly**: Interactive visualization (optional but recommended).
- **matplotlib**: Static plots (optional alternative).
- **pandas-datareader**: Fetch FRED economic data (may need `pip install pandas-datareader`).

### Handling Optional Dependencies

Some dependencies are optional:
- If `yfinance` is not installed, the simulator falls back to local cache or raises an error with instructions.
- If `plotly` or `matplotlib` is not installed, visualization functions will raise an error with install instructions.
- `defeatbeta_api` is a fallback data provider; not critical if unavailable.

---

## How to Use the Simulator

### Quick Start: Using `run_portfolio_batch.py`

1. **Place your data in `data/` folder:**
   - Add CSV files with columns `[Date, Adj Close]` for each ticker (e.g., `VTI.csv`, `BND.csv`).
   - Add FRED CSV files (e.g., `FRED_EFFR.csv`).
   - Alternatively, the first run will attempt to download missing data via yfinance.

2. **Edit `run_portfolio_batch.py`** to define portfolios:
   ```python
   portfolios = [
       {
           "name": "Classic 60/40",
           "assets": [
               SpotAssetConfig(id="VTI", ticker="VTI"),
               SpotAssetConfig(id="BND", ticker="BND"),
           ],
           "target_weights": {"VTI": 0.6, "BND": 0.4},
       },
   ]
   ```

3. **Run the batch:**
   ```bash
   python run_portfolio_batch.py
   ```

4. **Check results in `output/portfolio_metrics_summary.csv`.**

### Advanced: Using Notebooks (`main.ipynb` / `instructions.ipynb`)

1. **Open Jupyter Lab:**
   ```bash
   jupyter lab main.ipynb
   ```

2. **Define your simulation:**
   ```python
   from orchestration import (
       SimulationConfig, MarketDataConfig, PortfolioConfig, MonteCarloConfig,
       SpotAssetConfig, SyntheticLETFAssetConfig, evaluate_portfolio_from_config,
   )
   
   config = SimulationConfig(
       market=MarketDataConfig(
           start="2015-01-01",
           end="2023-12-31",
           fred_series="EFFR",
       ),
       assets=[
           SpotAssetConfig(id="VTI", ticker="VTI"),
           SyntheticLETFAssetConfig(
               id="VTI_3x",
               underlying_ticker="VTI",
               leverage=3.0,
               ter=0.0095,
               spread=0.003,
           ),
       ],
       portfolio=PortfolioConfig(
           target_weights={"VTI": 0.4, "VTI_3x": 0.6},
           initial_capital=100_000,
           capital_gains_tax_rate=0.20,
       ),
       monte_carlo=MonteCarloConfig(
           n_paths=1000,
           horizon_days=252 * 10,  # 10 years
           method="bootstrap",
           seed=42,
       ),
   )
   ```

3. **Run the simulation:**
   ```python
   result = evaluate_portfolio_from_config(config)
   print(result.metrics.summary)
   ```

4. **Visualize:**
   ```python
   from visuals import plot_spaghetti_paths
   plot_spaghetti_paths(
       result.portfolio.wealth_paths,
       n_sample=100,
       title="10-Year Portfolio Paths",
   )
   ```

---

## Code Structure & Best Practices

### Design Principles

1. **Modularity**: Each module has a single, well-defined responsibility. Functions are pure (no side effects) where possible.
2. **Validation**: Input validation occurs at module boundaries; invalid inputs raise descriptive `ValueError` or `TypeError`.
3. **Vectorization**: NumPy vectorization is used for MC path generation (fast). Portfolio simulation uses explicit loops (correct).
4. **Reproducibility**: Optional `seed` parameter ensures deterministic runs for debugging and publication.
5. **Comments**: Critical financial logic includes detailed docstrings and inline comments.

### Common Workflows

#### Scenario Analysis
Compare portfolios with different leverage:
```python
configs = [
    SimulationConfig(..., assets=[..., SyntheticLETFAssetConfig(..., leverage=1.0, ...)]),
    SimulationConfig(..., assets=[..., SyntheticLETFAssetConfig(..., leverage=2.0, ...)]),
    SimulationConfig(..., assets=[..., SyntheticLETFAssetConfig(..., leverage=3.0, ...)]),
]
results = [evaluate_portfolio_from_config(cfg) for cfg in configs]
```

#### Sensitivity to Tax Rate
```python
for tax_rate in [0.0, 0.10, 0.20, 0.37]:
    cfg = SimulationConfig(..., portfolio=PortfolioConfig(..., capital_gains_tax_rate=tax_rate))
    result = evaluate_portfolio_from_config(cfg)
    print(f"Tax Rate {tax_rate}: CAGR median = {result.metrics.summary.loc['CAGR', 'median']:.4f}")
```

#### Distributional Sensitivity
```python
# Compare bootstrap vs. parametric (normal)
cfg_bootstrap = SimulationConfig(..., monte_carlo=MonteCarloConfig(..., method="bootstrap"))
cfg_normal = SimulationConfig(..., monte_carlo=MonteCarloConfig(..., method="parametric", distribution="normal"))
cfg_student_t = SimulationConfig(..., monte_carlo=MonteCarloConfig(..., method="parametric", distribution="student_t", student_t_df=6.0))

result_boot = evaluate_portfolio_from_config(cfg_bootstrap)
result_norm = evaluate_portfolio_from_config(cfg_normal)
result_t = evaluate_portfolio_from_config(cfg_student_t)
```

---

## Interpretation Guide

### Understanding the Output

#### Summary Table
```
             mean      median        p5        p95
CAGR       0.0847     0.0923    0.0012    0.1523
Volatility 0.1823     0.1715    0.1289    0.2401
Max Drawdown -0.3420 -0.2987   -0.5621   -0.1823
Sharpe Ratio 0.4652   0.5389    0.0087    0.8234
...
```

**Interpretation**:
- **Median CAGR of 9.23%**: Half of paths delivered at least 9.23% annualized returns.
- **P5 CAGR of 0.12%**: In the worst 5% of scenarios, returns barely kept pace with inflation.
- **Median Max Drawdown of -29.87%**: Half of paths saw their portfolio fall to ~70% of peak value at worst.
- **Sharpe Ratio 0.54**: Risk-adjusted return of 54 basis points per unit of volatility.

#### Spaghetti Plot
A chart with 100 overlaid equity curves:
- **Tight bundle**: Low variance; most paths track closely (e.g., a portfolio of long bonds).
- **Wide spread**: High variance; outcomes range from strong to poor (e.g., a leveraged equity portfolio in high volatility).
- **Downward skew**: Many paths end below starting capital (potential ruin scenarios).

#### Probability of Ruin
```
Probability of Ruin: 0.023
```
**Interpretation**: 2.3% of scenarios saw the portfolio drop below 10% of initial capital. For a $100,000 portfolio, ruin means falling below $10,000.

---

## Limitations & Future Enhancements

### Current Limitations

1. **No Hedging**: The simulator does not model options, futures, or inverse ETFs for hedging.
2. **Zero Transaction Costs**: No bid-ask spreads or execution slippage (can be added).
3. **No Market Microstructure**: Assumes instant fills; doesn't model liquidity constraints.
4. **Stationarity Assumption**: Historical mean and covariance are treated as constant (real markets exhibit regime switching).
5. **Tax Simplifications**: 
   - Capital losses are not carried forward or used to offset gains.
   - Taxes are paid immediately in the rebalancing period (not deferred to year-end).
   - Assumes a single flat tax rate (no progressive brackets or alternative minimum tax).
6. **Rebalancing Determinism**: Rebalances occur on fixed calendar dates, not dynamically when drift exceeds tolerance.

### Potential Enhancements

1. **Regime Switching**: Parameterize different market regimes (bull, bear, crisis) with separate mean/covariance.
2. **Transaction Costs**: Add per-trade cost or percentage slippage.
3. **Dynamic Rebalancing**: Trigger rebalances when drift exceeds tolerance (not just on calendar dates).
4. **Tax Loss Harvesting**: Model strategic selling of losers to offset winners.
5. **Leverage Constraints**: Simulate margin calls or forced liquidation if leverage exceeds broker limits.
6. **Funding Cost Curves**: Model the term structure of funding costs (not just a single daily rate).
7. **Hedge Strategies**: Option overlays, put spreads, inverse ETF allocations.
8. **Forward-Looking Estimation**: Use GARCH or other time-series models for conditional volatility rather than historical averages.

---

## Output Files & Directory Structure

```
leveraged-etf-pf/
├── data/                          # Downloaded price/rate data (CSVs)
│   ├── VTI.csv
│   ├── BND.csv
│   ├── FRED_EFFR.csv
│   └── ...
├── output/                        # Simulation results
│   ├── portfolio_metrics_summary.csv  # Aggregated metrics
│   ├── all-weather/
│   │   ├── portfolio_metrics_summary.csv
│   │   └── wealth_paths.csv
│   ├── classic-60-40/
│   │   ├── portfolio_metrics_summary.csv
│   │   └── wealth_paths.csv
│   └── ...
├── constants.py                   # Shared constants
├── data_loader.py                 # Market data ingestion
├── letf_engine.py                 # LETF return calculation
├── montecarlo.py                  # Path generation
├── portfolio_sim.py               # Core simulation engine
├── metrics.py                     # Risk/return metrics
├── orchestration.py               # Workflow management
├── visuals.py                     # Plotting functions
├── run_portfolio_batch.py         # Batch runner
├── main.ipynb                     # Main notebook
├── instructions.ipynb             # Instructions notebook
├── download_data.ipynb            # Data download notebook
├── readme.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Troubleshooting

### Issue: "yfinance is not installed"
**Solution**: Install yfinance or use pre-cached CSV files in `data/`.
```bash
pip install yfinance
```

### Issue: "FRED data file not found"
**Solution**: Run `download_data.ipynb` to fetch FRED data, or manually add CSVs to `data/` folder.

### Issue: "Covariance matrix is not positive semi-definite"
**Solution**: The simulator adds small diagonal jitter to stabilize Cholesky decomposition. If this persists, check for zero-variance assets or missing data.

### Issue: Portfolio simulation is slow
**Solution**: This is expected. The simulator iterates path-by-path and day-by-day for exact PMC tracking. 
- Reduce `n_paths` or `horizon_days` for testing.
- Use parametric MC (faster than bootstrap).
- Run on a machine with multiple cores (future: add parallelization).

### Issue: NaN or inf values in metrics
**Solution**: Check that:
- Wealth paths are all positive (portfolio never ruined or crashes to zero).
- Returns contain no NaNs (check data_loader output).
- Risk-free rate is finite and reasonable.

---

## Contributing & Development

### Code Style
- Follow **PEP 8**. Format with `black` or `autopep8`.
- Use type hints where helpful.
- Add docstrings to public functions.

### Testing
- Unit tests would be valuable for `letf_engine`, `montecarlo`, and `metrics` modules.
- Integration tests for end-to-end workflows.

### Extending the Simulator
Common extensions:
1. Add new metrics (Calmar ratio, Information ratio, etc.).
2. Support different tax models (FIFO, specific ID, wash-sale rules).
3. Implement hedging overlays.
4. Add regime-switching scenarios.

---

## References & Further Reading

### Financial Concepts
- **Leverage & Volatility Drag**: Arnott & West (2016), "How Can 'Diversified' Leverage and Risk Parity Fail?"
- **Sharpe & Sortino Ratios**: Sortino & Price (1994), "Performance Measurement in a Downside Risk Framework."
- **Drawdown Analysis**: Chekhlov et al. (2005), "Drawdown Measure in Portfolio Optimization."
- **LETF Decay**: Gastineau (2003), "The Benchmark Index ETF Performance Problem."

### Implementation Details
- **Cholesky Decomposition**: Golub & Van Loan (2013), "Matrix Computations."
- **Bootstrap Methods**: Efron & Tibshirani (1993), "An Introduction to the Bootstrap."
- **PMC Tax Tracking**: Italian tax regulations; comparable to weighted-average cost basis.

### Python Libraries
- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Plotly Documentation: https://plotly.com/python/

---

## License & Attribution

This project is provided as-is for educational and research purposes. Please consult with a financial advisor or tax professional before making investment decisions based on simulator outputs.

---

## Summary: From Theory to Practice

This simulator embodies a rigorous, quantitative approach to portfolio analysis:

1. **Financial Realism**: Daily mechanics (LETF formula, borrowing costs, tax tracking) are modeled exactly, not approximated.
2. **Computational Precision**: Path-by-path iteration ensures PMC tax calculations are flawless.
3. **Distributional Insight**: Monte Carlo output provides a full picture of outcomes (mean, percentiles, tail risk) rather than a single "expected" return.
4. **Actionable Results**: Comprehensive metrics (CAGR, Sharpe, drawdown, ruin probability) enable informed comparisons across portfolio strategies.

By carefully controlling assumptions (seed, method, distribution) and interpreting results within their limitations, users can build confidence in long-term portfolio outcomes before deploying capital.

 
