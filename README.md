# Traffic Demand Prediction - Machine Learning Pipeline

This repository contains the source code for predicting traffic demand across various geographic hashes and times.

## Overview

Traffic demand forecasting is a complex spatiotemporal problem. This solution builds a robust machine learning pipeline utilizing **CatBoost** to capture non-linear interactions between location (`geohash`), time of day, and environmental variables (temperature, weather, road type).

## Repository Contents

- `data_engineering.py` - Script used for exploratory data analysis (EDA), generating feature statistics, and creating baseline visualizations to inform the modeling process.
- `solution.py` - The primary python script that processes the training data, engineers cyclical temporal features, trains the CatBoost model, and generates predictions for the test set.
- `Approach.md` - A detailed explanation of the feature engineering and modeling methodology.
- `submission.csv` - The generated target submission file containing predictions.

## Requirements

- Python 3.7+
- `pandas`
- `numpy`
- `catboost`

Install the required libraries using:
```bash
pip install pandas numpy catboost
```

## How to Run

1. Ensure your dataset files (`train.csv` and `test.csv`) are located in a `dataset` folder within this directory:
   ```
   dataset/
   ├── train.csv
   └── test.csv
   ```
2. (Optional) Run the data engineering and exploration script to view dataset statistics and generate distribution plots:
   ```bash
   python data_engineering.py
   ```

3. Run the primary solution script to train the model and generate predictions:
   ```bash
   python solution.py
   ```
   
   The script will:
   - Load the datasets.
   - Extract cyclical sine/cosine features from the timestamps.
   - Impute missing values for numeric and categorical variables.
   - Train a CatBoostRegressor on the processed training data.
   - Output `submission.csv` with the final demand predictions.

4. Submit `submission.csv` to the leaderboard.
