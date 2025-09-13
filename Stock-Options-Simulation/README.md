# Stock Options Simulation Project

This project explores simulation techniques for valuing **European stock options**.  
It reviews established option-pricing theory and compares multiple simulation approaches to the Black-Scholes analytical method.

## Description
- Evaluates **Monte Carlo** and **Binomial Tree** simulation models for pricing European options.
- Compares simulated option values to the **Black-Scholes** model to assess accuracy.
- Demonstrates how simulation can be a robust tool for estimating option value.

The analysis shows that the **Binomial Simulation** achieved the lowest percentage difference from Black-Scholes prices, highlighting its potential for accurate option valuation.

## How to Open and Read
- This project is a **single notebook**: `StockOptionsSimulationProject.ipynb`
- Place the notebook in any folder and open it with your preferred IDE, such as **Jupyter Notebook**, **JupyterLab**, or **Visual Studio Code**.
- Run or read through the cells to view the simulations, visualizations, and results.

## Requirements
The notebook uses common Python data-science and scientific libraries, including:
- `numpy`
- `pandas`
- `scipy` (including `scipy.stats` and `scipy.special.comb`)
- `matplotlib`
- `math` (Python standard library)

If these packages are not already installed, they can be added using your IDE’s package manager or a Python environment tool (e.g., `pip` or `conda`).

---
