import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Extracts spatiotemporal and categorical features.
    """
    df = df.copy()
    
    # 1. Temporal Features
    print("Extracting temporal features...")
    # Timestamp is in format H:M
    if 'timestamp' in df.columns:
        time_split = df['timestamp'].str.split(':', expand=True)
        df['hour'] = time_split[0].astype(int)
        df['minute'] = time_split[1].astype(int)
        
        # Cyclical temporal features
        df['time_in_mins'] = df['hour'] * 60 + df['minute']
        df['sin_time'] = np.sin(2 * np.pi * df['time_in_mins'] / 1440)
        df['cos_time'] = np.cos(2 * np.pi * df['time_in_mins'] / 1440)
        df.drop(['timestamp', 'time_in_mins'], axis=1, inplace=True)
        
    # 2. Handle Missing Values
    print("Handling missing values...")
    categorical_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    numeric_cols = ['NumberofLanes', 'Temperature']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    return df

def main():
    print("Loading datasets...")
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Preprocess
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    # Define features and target
    target_col = 'demand'
    features = [c for c in train_proc.columns if c not in ['Index', target_col]]
    
    X_train = train_proc[features]
    y_train = train_proc[target_col]
    X_test = test_proc[features]
    
    # Identify categorical features for CatBoost
    cat_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    
    print("Initializing CatBoost Regressor...")
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='R2',
        cat_features=cat_features,
        verbose=100,
        random_seed=42,
        task_type='CPU'
    )
    
    print("Training model...")
    # For a real pipeline, we'd use a validation set, but for final submission we train on all data
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    model.fit(train_pool)
    
    print("Generating predictions...")
    preds = model.predict(X_test)
    
    # Create submission
    print("Saving submission...")
    submission = pd.DataFrame({
        'Index': test['Index'],
        'demand': preds
    })
    
    # Ensure no negative demand
    submission['demand'] = submission['demand'].clip(lower=0)
    
    submission.to_csv('submission.csv', index=False)
    print("Saved model predictions to 'submission.csv'.")

if __name__ == '__main__':
    main()
