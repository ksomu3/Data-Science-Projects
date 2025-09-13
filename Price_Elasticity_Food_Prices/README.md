# Price Elasticity Regression Model

This project analyzes U.S. grocery price trends and consumer price sensitivity from **2019–2023**.  
It focuses on modeling how **income class** influences price elasticity—how responsive consumers are to price changes in common grocery items such as eggs, chicken, ice cream, and coffee.  
For more information about the results of the model, please report to the report. This price elasticity analysis was part of a larger project for school.

## Description
- Measures **price elasticity** using an R-based regression model, with income class included as interaction terms to reveal differences in sensitivity to price changes.

## Files
- `team126_priceelasticity_analysis.ipynb` – Main notebook containing the R code and analysis.
- `6242 Extract.csv` – Grocery transaction dataset used in the model.

## How to Open and Run
1. Open a Jupyter notebook - we will be using this IDE to run the language R.
2. Install R kernel in Jupyter notebook environment through install.packages("IRkernel")
3. Import the packages dplyr, ggplot2, MASS, car, rpart, randomForest, caret, and tidyverse.
4. Import the file "team126_priceelasticity_analysis.ipynb"
5. Please go the following link: https://drive.google.com/drive/folders/1y91-W1TM-aiv4VLy_4DqpRUSf_cI_b1u?usp=sharing and download the Grocery_Transaction_Data.zip to get the 6242 Extract.csv file
6. Save the ipynb and csv file in the same folder
7. In the ipynb file update the folder path that you are using. 
## Requirements

The notebook requires the following R packages:
```r
library(dplyr)
library(ggplot2)
library(MASS)
library(car)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(tidyverse)
```

Install any missing packages in R with:  
install.packages("package_name")
