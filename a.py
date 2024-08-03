import pickle

def load_best_params(filename='grid_search.pkl'):
    """Load the GridSearchCV object and extract the best parameters."""
    # Load the GridSearchCV object from the file
    with open(filename, 'rb') as file:
        grid_search = pickle.load(file)

    # Extract the best parameters
    best_params = grid_search.best_params_

    return best_params

# Example usage
best_params = load_best_params()
print("Best Parameters:", best_params)
import pickle

# Load the model from the file
with open('model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Check the type of the loaded object
print("Type of the loaded object:", type(model))
print("Content of the loaded object:", model)
