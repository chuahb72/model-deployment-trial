import joblib
import numpy as np

def run_app():
    # Load the frozen model
    model = joblib.load('house_model.pkl')
    
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