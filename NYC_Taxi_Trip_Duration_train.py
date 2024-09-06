import argparse
from NYC_Taxi_Trip_Duration_utils_data import load_data, preprocess_data, remove_outliers
from NYC_Taxi_Trip_Duration_utils_eval import train_model, evaluate_model, save_model

# Define the main function for training
def main(train_data_path, model_filename):
    # Data boundaries for filtering
    xlim = [-74.2591, -73.7004]
    ylim = [40.4774, 40.9176]

    # Load and preprocess training data
    train = load_data(train_data_path)
    train = preprocess_data(train, xlim, ylim)
    train = remove_outliers(train, 'trip_duration')

    # Train model
    X_train = train.drop(columns=['trip_duration'])
    y_train = train['trip_duration']

    best_model = train_model(X_train, y_train)
    evaluate_model(best_model, X_train, y_train)
    save_model(best_model, model_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,default = './Data/train.csv' , help='Path to training data')
    parser.add_argument('--model_filename', type=str, default='./models/model.pkl', help='Filename to save the model')
    args = parser.parse_args()
    main(args.train_data_path, args.model_filename)
