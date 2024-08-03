import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from sklearn.linear_model import Lasso

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def check_missing_data(train, validation):
    """Print the number of missing values in train and validation data."""
    print(f'Checking for missing data in train: {train.isna().sum().sum()}')
    print(f'Checking for missing data in validation: {validation.isna().sum().sum()}')

average_speeds  =[15.62, 16.32, 16.77, 17.30, 18.17, 19.19, 17.85, 14.47, 12.08, 11.73, 11.89, 11.56, 11.43, 11.54, 11.33, 11.31, 11.81, 11.86, 11.89, 12.71, 13.91, 14.40, 14.58, 15.13]
hour_to_speed = {hour: speed for hour, speed in enumerate(average_speeds)}

def add_average_hourly_speed(df, hour_to_speed):
    df['average_hourly_speed'] = df['pickup_hour'].map(hour_to_speed)
    return df

def filter_geographical_boundaries(df, xlim, ylim):
    """Filter data within specified geographical boundaries."""
    return df[
        (df.pickup_longitude > xlim[0]) & (df.pickup_longitude < xlim[1]) & 
        (df.dropoff_longitude > xlim[0]) & (df.dropoff_longitude < xlim[1]) &
        (df.pickup_latitude > ylim[0]) & (df.pickup_latitude < ylim[1]) & 
        (df.dropoff_latitude > ylim[0]) & (df.dropoff_latitude < ylim[1])
    ]

def apply_clustering(df, n_clusters=6):
    """Apply KMeans clustering to pickup and dropoff coordinates."""
    pickup_coordinates = df[['pickup_latitude', 'pickup_longitude']]
    dropoff_coordinates = df[['dropoff_latitude', 'dropoff_longitude']]

    pickup_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['pickup_cluster_label'] = pickup_kmeans.fit_predict(pickup_coordinates)

    dropoff_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['dropoff_cluster_label'] = dropoff_kmeans.fit_predict(dropoff_coordinates)
    
    return df

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371 * c
    return km

def calculate_bearing(row):
    """Calculate bearing between two points."""
    lat1 = np.radians(row['pickup_latitude'])
    lat2 = np.radians(row['dropoff_latitude'])
    diff_long = np.radians(row['dropoff_longitude'] - row['pickup_longitude'])
    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def calculate_trip_distance(df):
    """Add trip distance calculated using the Haversine formula and bearing."""
    df['trip_distance'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)
    df['bearing'] = df.apply(calculate_bearing, axis=1)
    return df

def extract_datetime_features(df):
    """Extract features from datetime."""
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['day_of_month'] = df['pickup_datetime'].dt.day
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['season'] = df['month'] % 12 // 3 + 1
    df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if 18 <= x <= 21 else 0)
    return df

def calculate_distance_to_center(df, center_coordinates):
    """Calculate the distance to city center."""
    df['distance_to_center'] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], center_coordinates[1], center_coordinates[0]), axis=1)
    return df

def calculate_distance_to_airport(df, airport_coordinates, column_name):
    """Calculate the distance to an airport."""
    df[column_name] = df.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], airport_coordinates[1], airport_coordinates[0]), axis=1)
    return df

def remove_outliers(df, column):
    """Remove outliers from a specified column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def add_time_features(df):
    """Add granular time features."""
    df['pickup_minute'] = df['pickup_datetime'].dt.minute
    df['pickup_minute_of_day'] = df['pickup_hour'] * 60 + df['pickup_minute']
    return df

def add_manhattan_distance(df):
    """Calculate the Manhattan distance between pickup and dropoff coordinates."""
    df['manhattan_distance'] = (
        np.abs(df['dropoff_longitude'] - df['pickup_longitude']) +
        np.abs(df['dropoff_latitude'] - df['pickup_latitude'])
    )
    return df

def add_interaction_features(df):
    """Add interaction features."""
    df['distance_speed_interaction'] = df['trip_distance'] * df['average_hourly_speed']
    return df

# Integrate new features into the preprocessing pipeline
def preprocess_data(df, xlim, ylim, hour_to_speed):
    df = filter_geographical_boundaries(df, xlim, ylim)
    df = apply_clustering(df)
    df = calculate_trip_distance(df)
    df = extract_datetime_features(df)
    df = add_average_hourly_speed(df, hour_to_speed)
    df = add_time_features(df)
    df = add_manhattan_distance(df)
    df = add_interaction_features(df)
    center_coordinates = (40.7580, -73.9855)  # Times Square coordinates
    df = calculate_distance_to_center(df, center_coordinates)
    jfk_coordinates = (40.6413, -73.7781)
    df = calculate_distance_to_airport(df, jfk_coordinates, 'distance_to_jfk')
    laguardia_coordinates = (40.7769, -73.8740)
    df = calculate_distance_to_airport(df, laguardia_coordinates, 'distance_to_laguardia')
    drop_columns = ['vendor_id', 'pickup_datetime', 'store_and_fwd_flag', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    df.drop(columns=drop_columns, inplace=True, axis=1)
    return df

def get_important_features(df, target_column, alpha=0.1):
    """Get important features using Lasso regression."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    feature_columns = numeric_df.columns[numeric_df.columns != target_column]
    X = numeric_df[feature_columns]
    y = numeric_df[target_column]
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    lasso_coef = pd.Series(lasso.coef_, index=feature_columns)
    important_features = lasso_coef[lasso_coef != 0].index.tolist()
    print("Important Features from Lasso:", important_features)
    return important_features


def save_important_features_data(train, validation, important_features, target_column, train_output_path, validation_output_path):
    """Save the datasets with only important features and the target variable."""
    train = train[important_features + [target_column]]
    validation = validation[important_features + [target_column]]
    train.to_csv(train_output_path, index=False)
    validation.to_csv(validation_output_path, index=False)

# Load and preprocess the train data
train_data_path = './Data/train.csv'
train = load_data(train_data_path)
xlim = [-74.2591, -73.7004]
ylim = [40.4774, 40.9176]
train = remove_outliers(train, 'trip_duration')
train = preprocess_data(train, xlim, ylim, hour_to_speed)

# Get the important features from the train data
important_features = get_important_features(train, 'trip_duration')

# Save the train data with important features
train_output_path = './Data/train_.csv'
train.to_csv(train_output_path, columns=important_features + ['trip_duration'], index=False)

# Assuming the validation data is loaded similarly and preprocessed
validation_data_path = './Data/val.csv'
validation = load_data(validation_data_path)
validation = preprocess_data(validation, xlim, ylim, hour_to_speed)

# Save the validation data with important features
validation_output_path = './Data/validation.csv'
validation.to_csv(validation_output_path, columns=important_features + ['trip_duration'], index=False)
