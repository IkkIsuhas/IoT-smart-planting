import pickle
import pandas as pd

try:
    with open('IoT_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    print("Model type:", type(model))
    
    # Try to see if it has feature names (scikit-learn models usually do if trained with DataFrames)
    if hasattr(model, 'feature_names_in_'):
        print("Feature names in:", model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        print("Number of features:", model.n_features_in_)
except Exception as e:
    print("Error loading model:", e)
