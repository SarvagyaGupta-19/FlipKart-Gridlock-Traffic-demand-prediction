# Approach: Spatiotemporal Modeling for Traffic Demand Prediction

## Overview
This repository contains the machine learning solution used to predict traffic demand for the given competition dataset. Traffic demand is highly cyclical and location-dependent, making it an excellent candidate for tree-based modeling combined with robust spatiotemporal feature engineering.

## Feature Engineering
To capture the underlying patterns in the traffic data, the following features were engineered from the raw data:

1. **Cyclical Temporal Features**:
   - The `timestamp` column (e.g., `14:30`) was parsed into discrete `hour` and `minute` components.
   - To account for the continuous cyclical nature of time (e.g., 23:45 is only 30 minutes away from 00:15), the time was converted into total daily minutes and then transformed using Sine and Cosine encodings.

2. **Categorical Processing**:
   - `geohash`: Used directly as a high-cardinality categorical feature. Tree-based models process these well using target statistics.
   - External metadata (`RoadType`, `LargeVehicles`, `Landmarks`, `Weather`) were treated as nominal categories. Missing values were explicitly imputed as `'Unknown'` to allow the model to learn patterns associated with missingness.

3. **Numeric Imputation**:
   - Continuous attributes such as `Temperature` and `NumberofLanes` were imputed with the dataset median to handle sparsity without skewing distributions.

## Modeling Strategy
The core of the solution relies on **CatBoost Regressor**, chosen for its state-of-the-art handling of categorical variables without requiring manual One-Hot or Target Encoding.

- **Algorithm**: CatBoost (Gradient Boosting on Decision Trees)
- **Loss Function**: RMSE (Root Mean Squared Error) optimized to maximize R² performance.
- **Hyperparameters**: 
  - Iterations: 1500
  - Learning Rate: 0.05
  - Depth: 8
  - L2 Regularization: 3
- **Categorical Features**: `geohash`, `RoadType`, `LargeVehicles`, `Landmarks`, `Weather` were explicitly passed to the `cat_features` parameter for optimized splitting.

## Conclusion
This approach prioritizes extracting cyclical time representations and leveraging CatBoost's inherent categorical strengths to accurately model the complex, non-linear spatiotemporal interactions driving traffic demand.
