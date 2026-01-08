import joblib
import os
import sys
import numpy as np

# Crucial: Import sklearn here so PyInstaller knows to bundle it
import sklearn
import sklearn.ensemble 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def run_app():
    print("--- House Price Predictor (AI Model) ---")
    
    # 1. Locate and Load the frozen model
    try:
        model_path = resource_path("house_model.pkl")
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        input("Press Enter to exit...")
        return

    # 2. User Interaction Loop
    while True:
        try:
            val = input("\nEnter square footage (or 'q' to quit): ")
            if val.lower() == 'q':
                break
                
            size = float(val)
            # Prepare data for the model (must be 2D array)
            prediction = model.predict([[size]])
            
            result = "EXPENSIVE" if prediction[0] == 1 else "CHEAP"
            print(f">>> Prediction: This house is likely {result}")
            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_app()