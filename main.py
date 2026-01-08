import joblib
import numpy as np
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def run_app():
    # Use the helper function to find the model file
    model_path = resource_path("house_model.pkl")
    model = joblib.load(model_path)
    
    print("--- House Price Predictor ---")
    try:
        size = float(input("Enter house square footage: "))
        prediction = model.predict([[size]])
        
        result = "EXPENSIVE" if prediction[0] == 1 else "CHEAP"
        print(f"Prediction: This house is likely {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_app()