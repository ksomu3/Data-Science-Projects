# Hotel Booking Cancellation Prediction

Hotels face a significant challenge in managing room reservations due to flexible cancellation policies.  
Many allow guests to cancel bookings up to 48 hours before arrival, leading to last-minute vacancies that are difficult to fill.  
This project analyzes hotel booking data to **predict whether a reservation will be canceled**, helping hotels optimize overbooking strategies, reduce financial risk, and improve occupancy rates.

## Description
- **Goal:** Identify key factors that influence cancellations and build models to predict them.
- **Models Used:**
  - **Logistic Regression** – Baseline classifier for initial feature significance testing.
  - **Random Forest** – Ensemble of decision trees trained on random subsets of data and features.
  - **XGBoost** – Gradient boosting model that builds trees sequentially to reduce bias and improve accuracy.
- **Model Tuning:** Applied **5-fold cross-validation** to optimize hyperparameters such as `mtry` (Random Forest) and learning rate, tree depth, and number of boosting rounds (XGBoost).
- **Outcome:** Compared models using test error to measure accuracy and select the best predictive approach.

## Files
- `Hw5_Appendix.ipynb` – R notebook with full data analysis and model implementation.
- `hotel_bookings.csv` – Dataset of hotel reservations and cancellations.

## How to Open and Run
1. Open a Jupyter notebook - we will be using this IDE to run the language R.
2. Install R kernel in Jupyter notebook environment through install.packages("IRkernel")
3. Download  `Hw5_Appendix.ipynb` and `hotel_bookings.csv` into the same folder
4. Install the R packages below if not already installed

## Requirements

The notebook requires the following R packages:
```r
library(dplyr)
library(ggplot2)
library(randomForest)
library(caret)
```

Install any missing packages in R with:  
install.packages("package_name")

