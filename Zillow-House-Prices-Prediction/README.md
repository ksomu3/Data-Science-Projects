# Zillow House Price Prediction – South Florida

This project analyzes real estate listings in **South Florida** to identify undervalued homes using machine learning models.  
It pulls listing and sale data using the **Zillow API**, preprocesses and engineers features, and compares performance across:
- Multiple Linear Regression (MLR)
- Neural Network
- Random Forest
- Gradient Boosting (XGBoost)

Each model outputs a **ranked list of homes** sorted by undervaluation.

---

## How to Run the Code

1. Ensure the following `.csv` files are in the **same directory** as your `.py` files:
   - `Doral_FL_ForSale.csv`
   - `Doral_FL_RecentlySold.csv`
   - (or corresponding files for Miami or Kendall)

2. Open one of the following model files:
   - `MLR.py`
   - `NeuralNetwork.py`
   - `RandomForest.py`
   - `GradientBoosting.py`

3. Update **lines 298–302** to change the `city` and `state` for prediction.
   - Available options:
     - `Doral, FL`
     - `Kendall, FL`
     - `Miami, FL`

4. Run the script. A new file will be created:
   - `City_State_ranked.csv` → e.g., `Doral_FL_ranked.csv`
   - Homes are sorted from **most undervalued to least** based on model prediction vs. list price

> ⚠️ API access has been deactivated. The models will use saved CSV files instead of calling Zillow’s API.

---

## Code Files

| Filename              | Description                         |
|-----------------------|-------------------------------------|
| `MLR.py`              | Multiple Linear Regression model    |
| `NeuralNetwork.py`    | Multi-layer Perceptron (Neural Net) |
| `RandomForest.py`     | Random Forest Regressor             |
| `GradientBoosting.py` | XGBoost Gradient Boosting model     |

Each file is standalone and trains a model based on city/state input.

---

## Output Format

Each model generates a CSV file named:
