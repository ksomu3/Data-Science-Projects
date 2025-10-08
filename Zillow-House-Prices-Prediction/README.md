Zillow House Price Prediction

Notes on Running the Code
	1.	Please ensure the attached CSV files (*_ForSale.csv and *_RecentlySold.csv) are in the same working directory as the .py files. These files are created automatically when the API is first called.
	2.	To run the models for a new city-state, edit lines 298–302 in the code. These lines allow you to specify the city and state for which you’d like to generate predictions.
	3.	Only the following city-state combinations can currently be used:
	•	Doral, FL
	•	Kendall, FL
	•	Miami, FL
Note: We unsubscribed from the API after completing our analysis to avoid additional costs. These models will still run using the saved CSVs from earlier API pulls.
	4.	Once a model is run, a new CSV file (e.g., Doral_FL_ranked.csv) will be created. This file ranks homes from most undervalued to least.
	5.	Running a different model on the same city will update the ranking using the new model’s predictions, overwriting the previous ranked file.

Code Files

We provide four Python scripts implementing the following predictive models:
	•	MLR New.py - Multiple Linear Regression
	•	Neural Network.py - Multi-layer Perceptron with ReLU activation
	•	Random Forest New.py - Random Forest Regressor
	•	Gradient Boosting.py - XGBoost Regressor

Each file is self-contained and will output predictions for a given city-state combination.

Output

The main output for each run is a ranked CSV file named:

City_State_ranked.csv

This file contains homes ranked by their “Undervalued Score” — the difference between the model’s predicted price and the listed price.

Each time a new model is run on the same city-state, the file is updated accordingly.

