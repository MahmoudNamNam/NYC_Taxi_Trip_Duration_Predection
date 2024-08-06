# NYC Taxi Trip Duration Prediction

## Project Overview

This project aims to predict the duration of taxi trips in New York City using a variety of regression techniques, including Polynomial Linear Regression, Ridge Regression, and Lasso Regression for feature extraction. The dataset used is the New York City Taxi Trip Duration Dataset, which contains detailed records of taxi trips including pickup and dropoff locations, times, and other related features.

## Project Structure

- `model_pipeline.py`: Main script to run the feature engineering, data preprocessing, model training, evaluation, and prediction.
- `test.py`: to make prediction and create Sample submission file.  
- `README.md`: Project documentation.
- `grid_search.pkl`: Saved GridSearchCV object for the best model.
- `model.pkl`: Trained model saved using joblib.
- `submission.csv`: Sample submission file.

## Dataset

The dataset includes the following key columns:

- `id`: Unique identifier for each trip
- `vendor_id`: ID of the taxi vendor
- `pickup_datetime`: Date and time when the trip started
- `dropoff_datetime`: Date and time when the trip ended
- `passenger_count`: Number of passengers
- `pickup_longitude`: Longitude where the trip started
- `pickup_latitude`: Latitude where the trip started
- `dropoff_longitude`: Longitude where the trip ended
- `dropoff_latitude`: Latitude where the trip ended
- `store_and_fwd_flag`: This flag indicates whether the trip record was sent to the vendor or held in vehicle memory before sending
- `trip_duration`: Duration of the trip in seconds (target variable)

## Feature Engineering

The following feature engineering techniques were applied to enrich the dataset:

1. **Geographical Boundaries Filtering**: Trips were filtered to stay within the specified geographical boundaries to remove erroneous data.
2. **Clustering**: KMeans clustering was applied to the pickup and dropoff coordinates to create cluster labels.
3. **Trip Distance Calculation**: The Haversine formula was used to calculate the great circle distance between pickup and dropoff points.
4. **Bearing Calculation**: The bearing between pickup and dropoff locations was calculated.
5. **Datetime Features Extraction**: Features like month, day of month, weekday, hour of day, and whether the trip occurred on a weekend or during rush hour were extracted.
6. **Average Hourly Speed**: An average hourly speed was mapped to the trips based on the hour of the day.
7. **Distance to Points of Interest**: Distances to the city center, JFK Airport, and LaGuardia Airport were calculated.
8. **Manhattan Distance**: The Manhattan distance between pickup and dropoff locations was calculated.
9. **Interaction Features**: Interaction features such as the product of trip distance and average hourly speed were created.
10. **Time Features**: Features like minute of the day and whether the trip happened during rush hour were added.

## Model Training

A Ridge Regression model with polynomial features was used for the final prediction. GridSearchCV was employed to tune the hyperparameters of the model.

### Training

The model was trained using the following metrics:

- **R² Score on Training Data**: 0.6729

### Validation

The model was evaluated on the validation set using the following metrics:

- **Validation RMSE**: 268.05
- **Validation MAE**: 203.69
- **Validation R² Score**: 0.6423

## Code Structure

The main functions and their purposes are outlined below:

1. **Data Loading and Preprocessing**
   - `load_data(file_path)`: Loads data from a CSV file.
   - `check_missing_data(train, validation)`: Checks for missing values in the train and validation data.
   - `preprocess_data(df, xlim, ylim, hour_to_speed, isTest=False)`: Integrates all preprocessing steps including feature engineering.

2. **Feature Engineering**
   - `add_average_hourly_speed(df, hour_to_speed)`: Adds average hourly speed to the dataframe.
   - `filter_geographical_boundaries(df, xlim, ylim)`: Filters data within specified geographical boundaries.
   - `apply_clustering(df, n_clusters=6)`: Applies KMeans clustering to pickup and dropoff coordinates.
   - `calculate_trip_distance(df)`: Adds trip distance and bearing to the dataframe.
   - `extract_datetime_features(df)`: Extracts features from datetime columns.
   - `calculate_distance_to_center(df, center_coordinates)`: Calculates the distance to city center.
   - `calculate_distance_to_airport(df, airport_coordinates, column_name)`: Calculates the distance to an airport.
   - `remove_outliers(df, column)`: Removes outliers from a specified column using the IQR method.
   - `add_time_features(df)`: Adds granular time features.
   - `add_manhattan_distance(df)`: Calculates the Manhattan distance between pickup and dropoff coordinates.
   - `add_interaction_features(df)`: Adds interaction features.

3. **Modeling and Evaluation**
   - `get_important_features(df, target_column, alpha=0.1)`: Extracts important features using Lasso regression.
   - `train_model(X_train, y_train)`: Trains the Ridge regression model with GridSearchCV.
   - `evaluate_model(model, X, y)`: Evaluates the model and prints cross-validated RMSE, MAE, and R² score.
   - `save_model(model, filename)`: Saves the trained model to a file.
   - `load_model(filename)`: Loads a model from a file.
   - `predict_and_save_submission(model, test_features, test_ids, filename)`: Predicts test data and saves the submission file.

## Instructions to Run the Code

1. **Install Dependencies**
   Make sure you have the required libraries installed. You can install them using:

   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

2. **Load Data**
   Use the `load_data` function to load your dataset.

3. **Preprocess Data**
   Apply the `preprocess_data` function to your dataset.

4. **Train the Model**
   Use the `train_model` function to train the Ridge regression model.

5. **Evaluate the Model**
   Evaluate the trained model using the `evaluate_model` function.

6. **Save the Model**
   Save the trained model using the `save_model` function.

7. **Load the Model**
   Load the saved model using the `load_model` function.

8. **Predict and Save Submission**
   Use the `predict_and_save_submission` function to generate predictions on the test set and save the results.

## Results

The final model achieved the following performance on the validation set:

- **Validation RMSE**: 268.05
- **Validation MAE**: 203.69
- **Validation R² Score**: 0.6423

The project demonstrates a comprehensive approach to feature engineering and model training for predicting taxi trip durations in New York City.

## References

- [NYC Taxi Trip Duration Dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

---
