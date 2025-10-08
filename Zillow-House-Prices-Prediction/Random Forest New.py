import requests
import json
import pandas as pd
import os
import time
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.inspection import PartialDependenceDisplay
import shap
from tqdm import tqdm


TRAINING_COLUMNS = []


def make_zillow_request(url, headers, params, retries=5, backoff=1.0):
    for attempt in range(retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            print(f"Rate limit hit. Retrying in {backoff} seconds...")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    print("Max retries reached.")
    return None


def fetch_property_details(zpid: str, api_key: str) -> Dict[str, Any]:
    url = "https://zillow-com1.p.rapidapi.com/property"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "zillow-com1.p.rapidapi.com"
    }
    querystring = {"zpid": zpid}

    response = make_zillow_request(url, headers=headers, params=querystring)
    if response is None:
        return None

    try:
        data = response.json()
        if isinstance(data, list):
            data = data[0]
        return data
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for ZPID {zpid}")
        return None



def extract_property_info(property_info: Dict[str, Any], sold_weight: float = 1.0, forsale_weight: float = 0.5) -> Dict[str, Any]:
    if not isinstance(property_info, dict):
        print(f"Unexpected data format: {type(property_info)}")
        return {}
    
    # Filter out excluded home types early
    excluded_home_types = {'MANUFACTURED', 'LOT','HOME_TYPE_UNKNOWN'}
    home_type = str(property_info.get("homeType", "N/A")).strip().upper()
    if home_type in excluded_home_types:
        return {}
    
    schools = property_info.get("schools", None)

    # Ensure schools is a list before proceeding
    if isinstance(schools, list):
        school_count = len(schools)
        school_max_rating = max([school.get("rating", 0) if school.get("rating") is not None else 0 for school in schools], default=0)
    else:
        school_count = 0
        school_max_rating = 0  # Or None if you prefer
    
    price_history = property_info.get("priceHistory", [])
    if isinstance(price_history, list):
        last_sold_price = next((event.get("price") for event in price_history if event.get("event") == "Sold"), None)
    else:
        last_sold_price = None

    resoFacts = property_info.get("resoFacts", {})
    nearbyHomes = property_info.get("nearbyHomes", [])
    current_home_type = property_info.get("homeType", "N/A")
    current_living_area = property_info.get("livingArea", "N/A")

    # Calculate houseAge based on the current year and yearBuilt
    current_year = datetime.now().year
    year_built = property_info.get("yearBuilt", "N/A")
    try:
        house_age = current_year - int(year_built) if year_built != "N/A" and isinstance(year_built, int) else "N/A"
    except ValueError:
        house_age = "N/A"

    # Calculate Nearby Home Metrics with Weighted Approach
    nearby_prices_sold = []
    nearby_prices_forsale = []
    nearby_sqft_prices_sold = []
    nearby_sqft_prices_forsale = []

    for home in nearbyHomes:
        home_type = home.get("homeType", None)
        status = home.get("status_type", "ForSale")  # Assuming status_type indicates if it's RecentlySold or ForSale

        if home_type == current_home_type:  # Only consider homes with the same type
            price = home.get("price")
            living_area = home.get("livingArea")
            
            if price and living_area:
                price_per_sqft = price / living_area
                
                if status == "RecentlySold":
                    nearby_prices_sold.append(price)
                    nearby_sqft_prices_sold.append(price_per_sqft)
                else:
                    nearby_prices_forsale.append(price)
                    nearby_sqft_prices_forsale.append(price_per_sqft)

    # Handle different cases for available data
    if nearby_prices_sold and nearby_prices_forsale:
        # Both datasets are available, apply weighted average
        all_prices = nearby_prices_sold + nearby_prices_forsale
        all_sqft_prices = nearby_sqft_prices_sold + nearby_sqft_prices_forsale
        weights = [sold_weight] * len(nearby_prices_sold) + [forsale_weight] * len(nearby_prices_forsale)

        weighted_avg_price = np.average(all_prices, weights=weights)
        weighted_avg_price_per_sqft = np.average(all_sqft_prices, weights=weights)
        
    elif nearby_prices_sold:
        # Only RecentlySold data is available
        weighted_avg_price = np.mean(nearby_prices_sold)
        weighted_avg_price_per_sqft = np.mean(nearby_sqft_prices_sold)
        
    elif nearby_prices_forsale:
        # Only ForSale data is available
        weighted_avg_price = np.mean(nearby_prices_forsale)
        weighted_avg_price_per_sqft = np.mean(nearby_sqft_prices_forsale)
        
    else:
        # No data available
        weighted_avg_price = None
        weighted_avg_price_per_sqft = None

    num_similar_homes = len(nearby_prices_sold) + len(nearby_prices_forsale)
    
    return {
        "zpid": property_info.get("zpid", "N/A"),
        "address": property_info.get("address", {}).get("streetAddress", "N/A") if isinstance(property_info.get("address"), dict) else "N/A",
        "city": property_info.get("address", {}).get("city", "N/A") if isinstance(property_info.get("address"), dict) else "N/A",
        "state": property_info.get("address", {}).get("state", "N/A") if isinstance(property_info.get("address"), dict) else "N/A",
        "zipcode": property_info.get("address", {}).get("zipcode", "N/A") if isinstance(property_info.get("address"), dict) else "N/A",
        "price": property_info.get("price", "N/A"),
        "homeType": property_info.get("homeType", "N/A"),
        "livingArea": property_info.get("livingArea", "N/A"),
        "lotAreaValue": property_info.get("lotAreaValue", "N/A"),
        "bedrooms": property_info.get("bedrooms", "N/A"),
        "bathrooms": property_info.get("bathrooms", "N/A"),
        "yearBuilt": property_info.get("yearBuilt", "N/A"),
        "houseAge": house_age,
        "monthlyHoaFee": property_info.get("monthlyHoaFee", "N/A"),
        "propertyCondition": resoFacts.get("propertyCondition", "N/A"),
        "architecturalStyle": resoFacts.get("architecturalStyle", "N/A"),
        "garageSpaces": resoFacts.get("garageParkingCapacity", "N/A"),
        "hasGarage": resoFacts.get("hasGarage", "N/A"),
        "hasPrivatePool": resoFacts.get("hasPrivatePool", "N/A"),
        "basementYN": resoFacts.get("basementYN", "N/A"),
        "parking": property_info.get("parking", "N/A"),
        "lastSoldPrice": last_sold_price,
        "schools_count": school_count,
        "schools_max_rating": school_max_rating,
        "daysOnZillow": resoFacts.get("daysOnZillow", "N/A"),
        "latitude": property_info.get("latitude", "N/A"),
        "longitude": property_info.get("longitude", "N/A"),
        "furnished": resoFacts.get("furnished", "N/A"),
        "storiesTotal": resoFacts.get("storiesTotal", "N/A"),
        "NearbyAvgPrice": weighted_avg_price,
        "NumSimilarHomesSold": num_similar_homes,
        "NearbyPricePerSqFt": weighted_avg_price_per_sqft
    }



def fetch_properties_by_city(city: str, state: str, api_key: str, status_type: str, max_results: int = 2000, max_pages: int = 40) -> pd.DataFrame:
    file_name = f"{city}_{state}_{status_type}.csv"
    if os.path.exists(file_name):
        print(f"Data for {city}, {state} ({status_type}) already exists. Loading from file.")
        return pd.read_csv(file_name)

    url = "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
    
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "zillow-com1.p.rapidapi.com"
    }

    data_list = []
    seen_zpids = set()
    total_results = 0
    excluded_home_types = {'MANUFACTURED', 'LOT','HOME_TYPE_UNKNOWN'}

    for page in tqdm(range(1, max_pages + 1), desc=f"Fetching {status_type} data for {city}, {state}"):
        querystring = {
            "location": f"{city}, {state}",
            "status_type": status_type,
            "page": str(page)
        }

        response = make_zillow_request(url, headers=headers, params=querystring)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break

        properties = response.json().get('props', [])
        if not properties:
            break

        for prop in properties:
            if total_results >= max_results:
                break

            zpid = prop.get("zpid")
            if not zpid or zpid in seen_zpids:
                continue
            seen_zpids.add(zpid)

            home_type = str(prop.get("homeType", "N/A")).strip().upper()
            if home_type in excluded_home_types:
                continue

            extracted_info = extract_property_info(prop)
            if not extracted_info:
                continue

            property_info = fetch_property_details(zpid, api_key)
            if property_info and isinstance(property_info, dict):
                detailed_home_type = str(property_info.get("homeType", "N/A")).strip().upper()
                if detailed_home_type in excluded_home_types:
                    continue

                additional_info = extract_property_info(property_info)
                if not additional_info:
                    continue
                extracted_info.update({k: v for k, v in additional_info.items() if v != "N/A"})

            data_list.append(extracted_info)
            total_results += 1

            time.sleep(0.2)

    df = pd.DataFrame(data_list)
    if not df.empty:
        df.to_csv(file_name, index=False)
        print(f"Data for {city}, {state} ({status_type}) saved to {file_name}")
    
    return df




def calculate_scaled_price(row):
    try:
        price_per_sqft = row.get('NearbyPricePerSqFt', None)
        living_area = row.get('livingArea', None)
        nearby_avg_price = row.get('NearbyAvgPrice', None)
        
        # Check if both price_per_sqft and living_area are valid numbers
        if isinstance(price_per_sqft, (int, float)) and isinstance(living_area, (int, float)):
            if price_per_sqft > 10000:  # Check if the price per sqft is above the threshold
                if isinstance(nearby_avg_price, (int, float)):  # Use NearbyAvgPrice if it exists
                    return nearby_avg_price
                else:
                    return None  # Return None if NearbyAvgPrice is not a valid number
            else:
                return price_per_sqft * living_area  # Calculate the scaled price normally
        
        return None  # Return None if inputs are invalid or missing
    except KeyError:
        return None  # Return None if keys are missing


def add_scaled_price_feature(property_df: pd.DataFrame):
    # Create the new column 'NearbyScaledPrices'
    property_df['NearbyScaledPrices'] = property_df.apply(calculate_scaled_price, axis=1)
    
    return property_df


# Key and City
api_key = "ef1968a55fmshcaf81f5db26a7bcp1a0ea6jsn3691041cb84a"
city = "Doral"
state = "FL"

# Fetch training data (Recently Sold Homes)
recently_sold_df = fetch_properties_by_city(city, state, api_key, status_type="RecentlySold")

# Filter out properties priced less than $30,000
recently_sold_df = recently_sold_df[pd.to_numeric(recently_sold_df['price'], errors='coerce') > 30000]

recently_sold_df = add_scaled_price_feature(recently_sold_df)

# Fetch test data (For Sale Homes)
forsale_df = fetch_properties_by_city(city, state, api_key, status_type="ForSale")

# Filter out properties priced less than $30,000
forsale_df = forsale_df[pd.to_numeric(forsale_df['price'], errors='coerce') > 30000]

forsale_df = add_scaled_price_feature(forsale_df)


def preprocess_training_data(recently_sold_df: pd.DataFrame):
    global TRAINING_COLUMNS

    # Define columns to drop
    drop_columns = [
        'zpid', 'address', 'city', 'state', 'yearBuilt', 'architecturalStyle', 
        'basementYN', 'longitude', 'latitude', 'furnished', 
        'storiesTotal', 'NumSimilarHomesSold', 'parking','lastSoldPrice','NearbyAvgPrice','NearbyPricePerSqFt'
    ]

    # Drop specified columns if they exist in the DataFrame
    recently_sold_df = recently_sold_df.drop(columns=[col for col in drop_columns if col in recently_sold_df.columns], errors='ignore')

    # Replace missing values in `lotAreaValue` using `livingArea`
    recently_sold_df.loc[recently_sold_df['lotAreaValue'].isna(), 'lotAreaValue'] = recently_sold_df['livingArea']

    # Drop columns that are 100% missing
    recently_sold_df = recently_sold_df.dropna(axis=1, how='all')

    # Separate features and target
    X = recently_sold_df.drop(columns=['price'], errors='ignore')
    y = recently_sold_df['price']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Convert all categorical columns to strings to avoid mixed types
    for col in categorical_cols:
        X[col] = X[col].astype(str)  # Ensure all categorical columns are strings

    # Preprocessing pipeline for numeric and categorical data
    numeric_transformer = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit the preprocessor on the training data only and transform it
    X_transformed = preprocessor.fit_transform(X)

    # Convert transformed data to a numpy array of type float64
    X_transformed = np.asarray(X_transformed, dtype=np.float64)

    # Save column names for later use
    categorical_features = preprocessor.transformers_[1][1].named_steps['onehotencoder'].get_feature_names_out(categorical_cols).tolist()
    TRAINING_COLUMNS = numerical_cols + categorical_features

    print("Training Columns Used:", TRAINING_COLUMNS)

    return X_transformed, y, preprocessor





def preprocess_test_data(forsale_df: pd.DataFrame, preprocessor):
    drop_columns = [
        'zpid', 'address', 'city', 'state', 'yearBuilt', 'architecturalStyle', 
        'basementYN', 'longitude', 'latitude', 'furnished', 
        'storiesTotal', 'NumSimilarHomesSold', 'parking','lastSoldPrice','NearbyAvgPrice','NearbyPricePerSqFt'
    ]

    forsale_df.loc[forsale_df['lotAreaValue'].isna(), 'lotAreaValue'] = forsale_df['livingArea']
    
    # Drop specified columns
    forsale_df = forsale_df.drop(columns=[col for col in drop_columns if col in forsale_df.columns], errors='ignore')


    # Preprocess test data using the fitted preprocessor
    X_test = preprocessor.transform(forsale_df)

    # Ensure the output is a numpy array
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    
    X_test = np.asarray(X_test, dtype=np.float64)
    
    return X_test




def detect_outliers(X, y, threshold=2.5):
    # Train a basic model to detect outliers
    model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    
    # Predict prices
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate standard deviation of residuals
    residual_std = np.std(residuals)
    
    # Define upper and lower limits for outlier detection
    lower_limit = -threshold * residual_std
    upper_limit = threshold * residual_std
    
    # Identify outliers
    outliers = (residuals < lower_limit) | (residuals > upper_limit)
    
    # Remove outliers
    X_cleaned = X[~outliers]
    y_cleaned = y[~outliers]

    return X_cleaned, y_cleaned



def plot_feature_importances(model, feature_names, top_n=10):
    # Extract feature importances from the model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a DataFrame to display the numerical summary
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    # Add rank based on importance
    feature_importance_df['Rank'] = range(1, len(feature_importance_df) + 1)
    
    # Display full DataFrame summary
    print("\nNumerical Summary of Feature Importances:")
    print(feature_importance_df)
    
    # Optionally display only the top N features
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance_df.head(top_n))
    
    # Plotting the top N features
    plt.figure(figsize=(15, 7))
    plt.title("Top Feature Importances", fontsize=16)
    plt.bar(range(top_n), feature_importance_df['Importance'].head(top_n), align="center")
    plt.xticks(range(top_n), feature_importance_df['Feature'].head(top_n), rotation=90, fontsize=12)
    plt.xlabel("Feature Name", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.tight_layout()
    plt.show()



def train_and_predict(recently_sold_df: pd.DataFrame, forsale_df: pd.DataFrame, n_splits: int = 5, forsale_weight: float = 0.5) -> pd.DataFrame:
    global TRAINING_COLUMNS

    # Preprocess the training data (RecentlySold)
    X_train_recently_sold, y_train_recently_sold, preprocessor = preprocess_training_data(recently_sold_df)
    
    # Remove outliers from training data
    X_train_recently_sold, y_train_recently_sold = detect_outliers(X_train_recently_sold, y_train_recently_sold)

    # Preprocess the ForSale data using the same preprocessor
    X_train_forsale, y_train_forsale = preprocess_test_data(forsale_df, preprocessor), forsale_df['price']
    
    # Remove NaN prices from ForSale dataset
    valid_indices = ~y_train_forsale.isna()
    X_train_forsale = X_train_forsale[valid_indices]
    y_train_forsale = y_train_forsale[valid_indices]

    # Combine RecentlySold and ForSale data
    X_combined = np.concatenate([X_train_recently_sold, X_train_forsale])
    y_combined = np.concatenate([y_train_recently_sold, y_train_forsale])

    # Assign weights: Full weight (1.0) for RecentlySold, Reduced weight for ForSale
    recently_sold_weights = np.ones(len(y_train_recently_sold))
    forsale_weights = np.full(len(y_train_forsale), forsale_weight)
    weights_combined = np.concatenate([recently_sold_weights, forsale_weights])

    # Train the model with cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    
    for train_index, test_index in kf.split(X_combined):
        X_train_cv, X_test_cv = X_combined[train_index], X_combined[test_index]
        y_train_cv, y_test_cv = y_combined[train_index], y_combined[test_index]
        weights_cv = weights_combined[train_index]

        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
        model.fit(X_train_cv, y_train_cv, sample_weight=weights_cv)
        
        y_pred_test = model.predict(X_test_cv)

        r2_scores.append(r2_score(y_test_cv, y_pred_test))
        mae_scores.append(mean_absolute_error(y_test_cv, y_pred_test))
        
    print(f"\nCross-Validation R² Scores: {r2_scores}")
    print(f"Average R² Score: {np.mean(r2_scores):.4f}")
    print(f"\nCross-Validation MAE Scores: {mae_scores}")
    print(f"Average MAE Score: {np.mean(mae_scores):.2f}")
    
    # Fit final model on entire training dataset
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    final_model.fit(X_combined, y_combined, sample_weight=weights_combined)
    
    # Plot Feature Importances
    plot_feature_importances(final_model, TRAINING_COLUMNS)

    # Visualize the first tree from the Random Forest
    plt.figure(figsize=(20, 10))
    plot_tree(final_model.estimators_[0], feature_names=TRAINING_COLUMNS, filled=True, max_depth=3)
    plt.title("Visualization of a Single Decision Tree (Depth = 3)")
    plt.show()

    # Predict on the combined training dataset
    y_pred_full = final_model.predict(X_combined)

    # Plot Actual vs. Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_combined, y_pred_full, alpha=0.5)
    plt.plot([min(y_combined), max(y_combined)], [min(y_combined), max(y_combined)], color='red', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Prices")
    plt.show()

    # Plot Residuals
    residuals = y_combined - y_pred_full
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_full, residuals, alpha=0.5)
    plt.hlines(0, min(y_pred_full), max(y_pred_full), color='red')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # Partial Dependence Plot (On first 3 features)
    try:
        PartialDependenceDisplay.from_estimator(final_model, X_combined, features=[0, 1, 2], feature_names=TRAINING_COLUMNS)
        plt.show()
    except Exception as e:
        print(f"Partial Dependence Plot failed: {e}")

    # Now preprocess the ForSale dataset using the same preprocessor
    X_test = preprocess_test_data(forsale_df, preprocessor)
    X_test = np.asarray(X_test, dtype=np.float64)

    # Predict prices for ForSale dataset
    y_pred_test = final_model.predict(X_test)
    
    # Add predictions and undervalued score to the original ForSale dataframe
    forsale_df['PredictedPrice'] = y_pred_test
    forsale_df['UndervaluedScore'] = forsale_df['PredictedPrice'] - forsale_df['price']

    # Generate SHAP values for the test dataset
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=TRAINING_COLUMNS)

    return forsale_df



def rank_properties(property_df: pd.DataFrame) -> pd.DataFrame:
    # Sort by Undervalued Score from most undervalued to least
    sorted_df = property_df.sort_values(by='UndervaluedScore', ascending=False)
    return sorted_df

# Train the model and predict prices for homes currently for sale
forsale_df_with_predictions = train_and_predict(recently_sold_df, forsale_df)

# Rank the properties based on undervalued score
ranked_df = rank_properties(forsale_df_with_predictions)

# Save the ranked dataframe to CSV
ranked_df.to_csv(f"{city}_{state}_ranked.csv", index=False)
print(f"Ranked properties saved to {city}_{state}_ranked.csv")




