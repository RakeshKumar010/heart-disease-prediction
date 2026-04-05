# app.py

import joblib
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask App
app = Flask(__name__)

# --- Load the Model and Scaler ---
# Both are loaded once at startup to avoid reloading on every request.
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    scaler = joblib.load('scaler.joblib')   # Step 1: Load the saved StandardScaler
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Define the features exactly as they appear in the CSV/training data
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    probability = None                      # Step 4: Will hold risk % passed to frontend

    if request.method == 'POST':
        if model is None:
            prediction_text = "ERROR: Prediction model is not loaded. Check server logs."
            return render_template('index.html', prediction_text=prediction_text, probability=probability)

        if scaler is None:
            prediction_text = "ERROR: Scaler is not loaded. Check server logs."
            return render_template('index.html', prediction_text=prediction_text, probability=probability)

        try:
            # 1. Collect all 13 features from the submitted form
            features = []
            for name in FEATURE_NAMES:
                value = float(request.form[name])
                features.append(value)

            # 2. Convert features into a NumPy array (2D: 1 sample × 13 features)
            input_data = np.array([features])

            # Step 2: Apply the scaler to normalize input exactly as during training
            input_data_scaled = scaler.transform(input_data)

            # 3. Make prediction on the scaled input
            prediction = model.predict(input_data_scaled)[0]

            # Step 3: Calculate risk probability using predict_proba
            # predict_proba returns [[prob_class_0, prob_class_1]]
            # Class 1 = Heart Disease present → that's our "risk" probability
            proba = model.predict_proba(input_data_scaled)[0]
            risk_score = proba[1]                           # Probability of class 1 (disease)
            probability = round(risk_score * 100, 1)        # Convert to percentage, e.g. 73.4

            # 4. Format the prediction text result
            if prediction == 1:
                prediction_text = (
                    "The model predicts: HIGH likelihood of Heart Disease. "
                    "Please consult a medical professional."
                )
            else:
                prediction_text = (
                    "The model predicts: LOW likelihood of Heart Disease. "
                    "Always consult a medical professional for diagnosis."
                )

        except ValueError:
            prediction_text = "ERROR: Please ensure all 13 fields are filled with valid numeric values."
        except Exception as e:
            prediction_text = f"An unexpected error occurred during prediction: {e}"

    # Step 4: Pass 'probability' to the template alongside prediction_text
    return render_template('index.html', prediction_text=prediction_text, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)