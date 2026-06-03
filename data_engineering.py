import pandas as pd
import numpy as np
import os
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_ENABLED = True
except ImportError:
    PLOTTING_ENABLED = False

def load_data():
    print("Loading raw training data for analysis...")
    df = pd.read_csv('dataset/train.csv')
    return df

def basic_exploration(df):
    print("\n--- Basic Dataset Info ---")
    print(f"Total Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Demand Statistics ---")
    print(df['demand'].describe())

def temporal_engineering(df):
    print("\n--- Engineering Temporal Features ---")
    df = df.copy()
    time_split = df['timestamp'].str.split(':', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['minute'] = time_split[1].astype(int)
    
    # Calculate hourly average demand
    hourly_demand = df.groupby('hour')['demand'].mean().reset_index()
    hourly_demand.rename(columns={'demand': 'avg_hourly_demand'}, inplace=True)
    
    df = pd.merge(df, hourly_demand, on='hour', how='left')
    return df, hourly_demand

def spatial_engineering(df):
    print("\n--- Engineering Spatial Features ---")
    df = df.copy()
    # Calculate average demand per geohash
    geo_demand = df.groupby('geohash')['demand'].agg(['mean', 'count']).reset_index()
    geo_demand.rename(columns={'mean': 'geohash_historical_demand', 'count': 'geohash_traffic_volume'}, inplace=True)
    
    df = pd.merge(df, geo_demand, on='geohash', how='left')
    return df

def generate_plots(df, hourly_demand):
    if not PLOTTING_ENABLED:
        print("\nMatplotlib/Seaborn not installed. Skipping plot generation.")
        return
        
    print("\n--- Generating Exploratory Plots ---")
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Demand Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['demand'].sample(10000), bins=50, kde=True)
    plt.title('Distribution of Traffic Demand (Sampled)')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.savefig('plots/demand_distribution.png')
    plt.close()
    
    # Plot 2: Hourly Demand Trend
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='avg_hourly_demand', data=hourly_demand, palette='viridis')
    plt.title('Average Traffic Demand by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Demand')
    plt.savefig('plots/hourly_demand_trend.png')
    plt.close()
    
    print("Plots saved to the 'plots/' directory.")

def main():
    print("Starting Data Engineering Pipeline...")
    df = load_data()
    
    # 1. Exploration
    basic_exploration(df)
    
    # 2. Feature Engineering & Analysis
    df_temp, hourly_demand = temporal_engineering(df)
    df_spatial = spatial_engineering(df_temp)
    
    # 3. Visualizations
    generate_plots(df_spatial, hourly_demand)
    
    print("\nData Engineering and Exploratory Analysis Complete.")
    print("Proceeding to use 'solution.py' for final model training which incorporates these insights.")

if __name__ == "__main__":
    main()
