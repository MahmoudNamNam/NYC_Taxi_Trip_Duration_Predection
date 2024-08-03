import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model_pipeline import preprocess_data, load_data, hour_to_speed

def create_submission(predictions, output_path='submission.csv'):
    """Save the predictions to a CSV file."""
    submission = pd.DataFrame({
        'id': range(len(predictions)), 
        'trip_duration': predictions
    })
    submission.to_csv(output_path, index=False)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using test data."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')

def main():
    # Load model
    model = joblib.load('model.pkl')
    feature_columns = ['passenger_count', 'pickup_cluster_label', 'dropoff_cluster_label', 'trip_distance', 'bearing', 'month', 'day_of_month', 'weekday', 'is_weekend', 'season', 'is_rush_hour', 'average_hourly_speed', 'pickup_minute', 'pickup_minute_of_day', 'manhattan_distance', 'distance_speed_interaction', 'distance_to_center', 'distance_to_jfk', 'distance_to_laguardia']
    
    # Load and preprocess test data
    test_df = load_data('Data/test.csv')
    xlim = [-74.03, -73.75]
    ylim = [40.63, 40.85]
    processed_test_df = preprocess_data(test_df, xlim, ylim, hour_to_speed,True)
    processed_test_df = processed_test_df[feature_columns]
    # Load and preprocess ground truth values
    y_test = load_data('Data/sample_submission.csv')['trip_duration'] 
    
    # Evaluate the model
    evaluate_model(model, processed_test_df, y_test)
    
    # Make predictions
    predictions = model.predict(processed_test_df)
    
    # Create submission file
    create_submission(predictions)

if __name__ == '__main__':
    main()

