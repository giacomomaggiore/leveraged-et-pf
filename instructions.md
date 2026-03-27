# Technical Specifications: Leveraged ETF (LETF) Portfolio Simulator

## 1. Project Overview
The goal of this project is to build a modular, highly accurate quantitative simulation framework in Python to analyze the long-term viability of Daily Leveraged ETFs (LETFs) in passive portfolios. The system must account for volatility drag, dynamic borrowing costs (swap rates), exact capital gains taxation (tracking average cost basis), and path dependency.

## 2. Tech Stack & Architecture
*   **Main Interface:** Jupyter Notebook (`main.ipynb`) for execution, orchestration, and visualization.
*   **Helper Modules:** External `.py` files containing classes/functions to keep the notebook clean.
*   **Data Handling:** `numpy`, `pandas`.
*   **Data Sources:** `yfinance` (equities/bonds), `pandas-datareader` (FRED for swap/risk-free rates).
*   **Visualization:** `plotly` (preferred for interactive exploration of MC paths) or `matplotlib`/`seaborn`.

### File Structure Strategy
*   `data_loader.py`: Data ingestion and cleaning.
*   `letf_engine.py`: Calculation of synthetic LETF daily returns.
*   `montecarlo.py`: Generation of simulated market paths.
*   `portfolio_sim.py`: Core iterative loop for portfolio valuation, rebalancing, and tax calculation.
*   `metrics.py`: Risk and return calculations.
*   `visuals.py`: Plotting functions.

---

## 3. Module Specifications & Financial Requirements

### 3.1 Data Acquisition (`data_loader.py`)
*   **Equities/Bonds:** Use `yfinance` to download daily adjusted close prices (`Adj Close`). Calculate daily simple returns. Keep modular (accept any list of tickers).
*   **Borrowing Costs/Swap Rates:** Use `pandas-datareader` to query the FRED database (e.g., SOFR, Fed Funds, or historical proxies).
*   **Data Alignment:** Use a **trading-days-only** datetime index (no weekend/holiday rows). Align all series on trading days; forward-fill only where needed on trading-day gaps after merge.
*   **Rate Conversion Convention:** Convert annualized rates to daily using **252 trading days**.

### 3.2 LETF Engine (`letf_engine.py`)
Must construct synthetic daily returns for LETFs based on the underlying index and a dynamic borrowing cost.
*   **Inputs:** Underlying daily return ($R_{i,t}$), Leverage Factor ($L$), Annual TER, Daily Borrowing Rate ($Rate_t$), Spread.
*   **Units Convention:** TER, borrowing rate, and spread are all decimal rates (e.g., 0.0095 for 0.95%).
*   **Formula:** The daily return of the LETF ($R_{L,t}$) MUST be calculated as:
    $$R_{L,t} = L \cdot R_{i,t} - \left[ \frac{TER}{252} + (L-1) \cdot \frac{Rate_t + Spread}{252} \right]$$
*   **Trading-Day Application:** Cost terms are applied only on trading days.
*   **Output:** A pandas Series of synthetic daily returns for the LETF.

### 3.3 Monte Carlo Engine (`montecarlo.py`)
Must support two distinct simulation methods for generating `N` scenarios of length `T` days.
*   **Method A (Parametric - Synthetic):** 
    *   Calculate historical mean returns and the Covariance Matrix.
    *   Use Cholesky decomposition to generate correlated random daily returns. 
    *   Allow switching between Normal and t-Student distributions.
*   **Method B (Historical Bootstrapping - DEFAULT):**
    *   Sample *entire rows* (days) from the historical aligned dataframe with replacement.
    *   This preserves the exact historical correlation matrix and the non-normal "fat tails" (kurtosis) organically.
*   **Reproducibility:** Include an optional `seed` parameter for deterministic simulations.
*   **Output:** Return a 3D numpy array with shape **(paths, time, assets)**.

### 3.4 Portfolio Simulator (`portfolio_sim.py`)
This is the core engine. Due to exact tax tracking and path dependency, this module must simulate the portfolio day-by-day (or year-by-year for rebalancing events) using a programmatic loop (the "exact but slow" requirement).
*   **Inputs:** Simulated returns, initial capital, target weights, rebalancing frequency (Annual), tolerance bands (e.g., 5%), capital gains tax rate (default: 0.0).
*   **Position Granularity:** Fractional shares are allowed.
*   **Trading Frictions:** Transaction costs and slippage are assumed zero by default.
*   **Portfolio Overlay Policy:** No hedge overlay is used in this prototype.
*   **Rebalancing Logic:**
    *   Triggered **Annually** (e.g., every 252 trading days).
    *   Check if any asset's current weight deviates from its target weight by more than the `tolerance_band`.
    *   If NO: Do nothing (let it ride).
    *   If YES: Rebalance the *entire* portfolio back to exact target weights.
*   **Taxation Logic (Prezzo Medio di Carico - PMC):**
    *   The system must track the number of shares ($Q$) and the exact Average Cost Basis ($PMC$) for each asset.
    *   **Buy Event:** Update PMC.
        $$PMC_{new} = \frac{PMC_{old} \cdot Q_{old} + P_{buy} \cdot Q_{buy}}{Q_{old} + Q_{buy}}$$
    *   **Sell Event:** Calculate Realized Capital Gains.
        $$Gain = (P_{sell} - PMC_{old}) \cdot Q_{sell}$$
    *   **Rebalance Execution Sequence (MANDATORY):** execute in this order: **sell -> pay taxes -> buy**.
    *   **Tax Payment:** If $Gain > 0$, subtract $Gain \cdot TaxRate$ from the portfolio's cash pool *before* reinvesting during the rebalance. (Note: ignore capital losses compensation for this prototype, assume taxes are paid immediately on profits).

### 3.5 Risk & Return Metrics (`metrics.py`)
Evaluate the simulated paths and output aggregated statistics (mean, median, 5th percentile, 95th percentile) for:
1.  **CAGR:** Compound Annual Growth Rate.
2.  **Max Drawdown:** Deepest peak-to-trough decline.
3.  **Drawdown Duration (Recovery Time):** Maximum days spent below the High-Water Mark.
4.  **Sharpe & Sortino Ratios:** Risk-adjusted returns (using the simulated risk-free rate).
5.  **Probability of Ruin:** Percentage of MC paths where the portfolio drops below a critical threshold (e.g., > 90% loss of initial capital).
6.  **Ulcer Index:** To measure the depth and duration of drawdowns.

Implementation standard for metrics: follow quantitative finance best practices with explicit, consistent annualization and downside handling.
*   **Return Convention:** Use arithmetic daily portfolio returns for Sharpe/Sortino and geometric compounding for CAGR.
*   **Annualization:** Use 252 trading days for annualized return and volatility scaling.
*   **Sharpe Ratio:**
    $$Sharpe = \frac{\mu(r_p - r_f)}{\sigma(r_p - r_f)} \cdot \sqrt{252}$$
*   **Sortino Ratio:**
    $$Sortino = \frac{\mu(r_p - r_f)}{\sigma_{down}(r_p - r_f)} \cdot \sqrt{252}$$
    where $\sigma_{down}$ is computed using only negative excess-return observations.
*   **Max Drawdown:** Compute from pathwise cumulative wealth versus rolling high-water mark.
*   **Drawdown Duration:** Count consecutive trading days below high-water mark; recovery occurs only when previous peak is reached or exceeded.

### 3.6 Visualization (`visuals.py`)
Provide functions to plot:
1.  **Spaghetti Plot:** A sample of 50-100 equity curves from the MC simulation.
2.  **Distribution of Terminal Wealth:** Histogram showing the spread of final portfolio values across all scenarios.
3.  **Drawdown Chart:** Median and worst-case drawdown curves over time.

## 4. Implementation Notes for the AI Developer
*   Write modular, heavily commented, and PEP-8 compliant code.
*   Prioritize `numpy` vectorization where possible (e.g., generating the MC paths), but strictly adhere to the iterative loop for the Portfolio Simulator to ensure PMC tracking is mathematically flawless.
*   Do NOT hardcode tickers or specific model portfolios. Ensure the `Portfolio` class accepts arbitrary dictionaries of `{'Asset_Name': Target_Weight}`.