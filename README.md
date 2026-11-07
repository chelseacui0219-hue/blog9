
# Blog Post 9 — Predicting Stock Returns with Gradient Boosting

## Project Overview

This repository contains the complete analysis and supporting materials for Blog Post 9, which investigates whether regularization improves the predictive accuracy and investment performance of an XGBoost stock-return forecasting model.

The project builds on an instructor-provided R script and extends the analysis using Python in Google Colab.

**Research question:** Does a more regularized XGBoost model forecast stock returns more accurately, and does improved accuracy translate into higher portfolio returns?

## Repository Structure

```text
Blog-Post-9-XGBoost/
├── README.md
├── index.qmd
├── coding/
│   ├── analysis.ipynb
│   └── original_teacher_code.R
└── output/
    ├── figures/
    │   ├── forecast_errors_final.png
    │   └── portfolio_returns_final.png
    └── tables/
        ├── table9_forecast_with_zero.csv
        └── table10_cost_sensitivity.csv
```

### File Descriptions

- `index.qmd`: Quarto blog post containing the research question, methodology, results, interpretation, and conclusion. The article contains no executable code chunks.
- `coding/analysis.ipynb`: Complete Python notebook for data retrieval, feature engineering, model training, prediction, backtesting, and output generation.
- `coding/original_teacher_code.R`: Original classroom R script used as the starting point for the project.
- `output/figures/`: Static figures illustrating prediction errors and investment performance.
- `output/tables/`: Exported CSV files containing final forecasting and transaction-cost results.

## Data

**Source:** Yahoo Finance, accessed using the Python `yfinance` package.

**Stock universe:** Approximately 95 selected U.S. large-cap stocks, subject to data availability and quality filters.

**Period:** January 2022 to October 2026, with the download end date set to November 1, 2026.

**Benchmark:** SPY, an ETF tracking the S&P 500.

**Risk-free rate:** Treasury-bill yield proxy from the `^IRX` series.

### Prediction Target

The target is the adjusted-price return between the close at trading day t+1 and the close at t+20, representing 19 trading-day return intervals.

### Predictors

Features include:

- Historical returns
- Momentum indicators
- Volatility measures
- Price and trading-volume information
- Market-relative returns

The exact feature definitions are documented in the analysis notebook.

## Models

Four forecasting approaches are compared.

### 1. Historical Mean

Predicts future returns using each stock's average return estimated from the training sample.

### 2. Original XGBoost

Approximates the classroom specification:

- 300 boosting trees
- Maximum tree depth of 5
- Learning rate of 0.05

### 3. Regularized XGBoost

Uses stronger constraints on model complexity. Hyperparameters are selected through walk-forward validation using training-period observations only.

### 4. Zero-Return Forecast

Predicts zero for every observation. This serves as a simple forecasting benchmark rather than an investment strategy.

## Experimental Design

Training observations are drawn from the period before 2024. Out-of-sample evaluation begins in 2024.

Observations whose target windows cross the training/test boundary are excluded from training to reduce look-ahead bias.

All forecasting methods are evaluated on the same 65,645 test observations.

Prediction metrics include:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson correlation between predicted and realized returns

## Portfolio Backtesting

At each monthly signal date:

1. Rank stocks by predicted future return.
2. Select the ten highest-ranked stocks.
3. Assign weights proportional to positive predicted returns.
4. Use equal weights if valid proportional weights cannot be constructed.
5. Enter at the close of t+1 and exit at the close of t+20.

The backtest contains 34 non-overlapping holding periods.

The portfolios are compared with SPY and an equal-weight stock-universe benchmark over matched dates.

### Transaction Costs

An illustrative cost of 0.1% is applied to each purchase and sale.

For each holding period:

Net return = (1 - 0.001)^2 × (1 + gross return) - 1

This assumes complete liquidation and repurchase in every period, including for SPY. It is a sensitivity analysis rather than a realistic turnover-based cost model.

## Results

### Forecasting Accuracy

| Model | RMSE | MAE | Pearson Correlation |
|---|---:|---:|---:|
| Historical mean | 0.0891 | 0.0647 | 0.0473 |
| Original XGBoost | 0.0934 | 0.0682 | 0.0307 |
| Regularized XGBoost | **0.0884** | **0.0640** | **0.0719** |
| Zero return | 0.0895 | 0.0644 | N/A |

Regularization reduced RMSE by approximately 5.4% relative to the original model.

However, the highest prediction correlation was only 0.0719, indicating that the forecasting signal remained weak.

### Portfolio Performance

| Strategy | Gross Cumulative Return | Net Cumulative Return | Net Sharpe |
|---|---:|---:|---:|
| Historical mean | 127.64% | 112.67% | 1.20 |
| Original XGBoost | **128.98%** | **113.92%** | **1.57** |
| Regularized XGBoost | 105.53% | 92.01% | 1.50 |
| Equal-weight universe | 61.93% | 51.28% | 1.10 |
| SPY | 55.97% | 45.71% | 0.90 |

Returns are cumulative over 34 holding periods, not annualized.

Sharpe ratios use excess returns over a Treasury-bill proxy.

## Main Findings

The regularized XGBoost model achieved the lowest prediction errors, but the original model generated stronger backtested investment returns.

This difference highlights the distinction between minimizing forecasting errors and identifying profitable investment opportunities.

The historical-mean strategy also performed competitively, suggesting that strong portfolio returns cannot automatically be attributed to model complexity.

## How to Reproduce

### Step 1: Open the Notebook

Open `coding/analysis.ipynb` in Google Colab.

### Step 2: Install Dependencies

Run the notebook's setup cells to install the required Python packages, including `yfinance` and `xgboost`.

### Step 3: Run the Analysis

Execute the notebook from top to bottom.

The workflow:

1. Downloads market data.
2. Constructs features and future-return targets.
3. Creates training and test samples.
4. Trains forecasting models.
5. Evaluates predictions.
6. Constructs investment portfolios.
7. Performs backtesting.
8. Calculates transaction-cost sensitivity.
9. Exports tables and figures.

### Step 4: Save Results

Copy the generated figures and tables into the repository's `output/` directory.

The notebook writes its results to its configured output directory. The final files in this repository represent the reported experimental results.

### Step 5: Render the Blog

The `index.qmd` article contains no executable R or Python code.

It can be rendered using Quarto or incorporated into an existing Quarto website.

## Limitations

**Survivorship bias:** The selected stock universe may favor companies that remained successful during the sample period.

**Limited test period:** Thirty-four holding periods are insufficient to establish stable long-term profitability.

**Overlapping observations:** Daily stock-return targets overlap and should not be interpreted as independent observations.

**Backtest assumptions:** The simulation does not fully account for liquidity constraints, market impact, taxes, or slippage.

**Transaction-cost assumptions:** SPY is charged a hypothetical round-trip cost every period, unlike a conventional buy-and-hold investment.

**Reproducibility:** Yahoo Finance data can be revised, and differences in software versions may affect rerun results. The exact historical raw-data snapshot and full run metadata are not included.

## Result Consistency Note

An earlier run reported approximately 142.22% cumulative return for the original XGBoost strategy.

After a numerical-stability adjustment to portfolio weighting, the latest exported results reported 128.98% gross cumulative return and 113.92% net cumulative return.

The repository uses the latest exported results. A period-by-period comparison was not available to establish the exact cause of the difference.

## Conclusion

The experiment demonstrates that improving prediction accuracy does not necessarily improve investment performance.

Regularization reduced out-of-sample forecasting errors, but the original XGBoost strategy achieved higher simulated portfolio returns.

The findings emphasize the importance of evaluating financial machine learning models using both statistical metrics and carefully qualified investment backtests.

## Attribution

- Market data: Yahoo Finance
- Original classroom experiment: Instructor-provided R script
- Extended analysis: Python, XGBoost, Google Colab
- Presentation: Quarto
