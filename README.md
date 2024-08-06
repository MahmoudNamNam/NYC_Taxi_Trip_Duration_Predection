# NYC Taxi Trip Duration Prediction

## Project Overview

This project aims to predict the duration of taxi trips in New York City using machine learning models. The primary goal is to leverage various feature engineering techniques and regression models to achieve accurate predictions.

## Data

The dataset used for this project is the New York City Taxi Trip Duration dataset, which includes information such as pickup and dropoff times, locations, and other trip-related features.

## Environment Setup

### Dependencies

The following libraries are required to run the code:

- numpy
- pandas
- scikit-learn
- joblib
- warnings

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn joblib
```

## Project Structure

- `main.py`: Main script to run the data preprocessing, model training, evaluation, and prediction.
- `utils.py`: Utility functions for feature engineering, model training, and evaluation.
- `README.md`: Project documentation.
- `grid_search.pkl`: Saved GridSearchCV object for the best model.
- `final_model.pkl`: Trained model saved using joblib.
- `submission.csv`: Sample submission file.

## Data Preprocessing

The preprocessing steps include:

1. Loading the data.
2. Filtering data within specified geographical boundaries.
3. Applying clustering to pickup and dropoff coordinates using KMeans.
4. Calculating trip distance using the Haversine formula and bearing.
5. Extracting datetime features such as month, day, hour, and whether the trip was on a weekend or during rush hour.
6. Calculating the distance to city center, JFK airport, and LaGuardia airport.
7. Adding interaction features like distance-speed interaction.
8. Removing outliers based on trip duration.
9. Adding Manhattan distance between pickup and dropoff coordinates.

## Feature Engineering

- Average hourly speed based on pickup hour.
- Clustering of pickup and dropoff coordinates.
- Trip distance and bearing using the Haversine formula.
- Distance to city center and airports.
- Manhattan distance and interaction features.

## Models

### Lasso Regression

Used to extract important features with Lasso regression.

### Ridge Regression with Polynomial Features

Trained using a pipeline with feature scaling and polynomial features.

## Training and Evaluation

- Ridge regression model is trained with GridSearchCV for hyperparameter tuning.
- Cross-validated RMSE, MAE, and R² scores are computed.

### Model Performance

#### Training
- R² score: 0.6729

#### Validation
- RMSE: 268.05
- MAE: 203.69
- R² score: 0.6423

## Usage

### Train the Model
```python
from utils import load_data, preprocess_data, train_model, evaluate_model, save_model

# Load and preprocess the data
train_data = load_data('train.csv')
train_data = preprocess_data(train_data, xlim, ylim, hour_to_speed)

# Split the data into features and target
X_train = train_data.drop(columns=['trip_duration'])
y_train = train_data['trip_duration']

# Train and save the model
best_model = train_model(X_train, y_train)
evaluate_model(best_model, X_train, y_train)
save_model(best_model, 'final_model.pkl')
```

### Predict and Save Submission
```python
from utils import load_model, preprocess_data, predict_and_save_submission

# Load the model and test data
model = load_model('final_model.pkl')
test_data = load_data('test.csv')
test_data = preprocess_data(test_data, xlim, ylim, hour_to_speed, isTest=True)

# Make predictions and save submission
predict_and_save_submission(model, test_data, test_data['id'], 'submission.csv')
```

## Conclusion

This project demonstrates the use of extensive feature engineering and machine learning models to predict NYC taxi trip durations. The combination of geographical, temporal, and interaction features, along with polynomial and ridge regression, provides a robust approach for trip duration prediction.

## References

- [NYC Taxi Trip Duration Dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)

---

This README provides a comprehensive overview of the project, guiding users through setup, data preprocessing, model training, and evaluation steps.