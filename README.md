# 🛠️ Heart Disease Prediction System

A machine learning web application that predicts the likelihood of heart disease based on 13 clinical markers, built with Flask, Scikit-learn, and a clean violet-themed UI.

---

## 📁 Project Structure

```
heart-disease-predictor/
├── app.py                        # Flask backend — routes, prediction logic
├── model.py                      # Training script — ML models + saves joblib files
├── model.joblib                  # Saved best model (auto-selected by ROC-AUC)
├── scaler.joblib                 # Saved StandardScaler (fitted on training data)
├── heart_cleveland_upload.csv    # Dataset (Cleveland Heart Disease)
└── templates/
    └── index.html                # Frontend UI — form + result display
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn joblib
```

---

## 🚀 Usage

### Step 1 — Train the model
Run `model.py` **first** to generate `model.joblib` and `scaler.joblib`:
```bash
python model.py
```
Expected output:
```
Scaler saved as scaler.joblib
🔹 Logistic Regression Results ...
🔸 Random Forest Results ...
Best model (Random Forest) saved as model.joblib
✅ Both files saved successfully
```

### Step 2 — Start the Flask app
```bash
python app.py
```

### Step 3 — Open in browser
```
http://127.0.0.1:5000
```

Fill in the 13 clinical fields and click **GET PREDICTION** to see the result with risk percentage.

---

## 🧠 Machine Learning Pipeline

| Step | Details |
|------|---------|
| **Dataset** | Cleveland Heart Disease (`heart_cleveland_upload.csv`) |
| **Target** | `condition` → renamed to `target` (0 = No Disease, 1 = Disease) |
| **Features** | 13 clinical markers (age, sex, cp, trestbps, chol, etc.) |
| **Split** | 80% train / 20% test, stratified, `random_state=42` |
| **Scaling** | `StandardScaler` — fit on train, transform on test & live input |
| **Models** | Logistic Regression + Random Forest (100 trees) |
| **Best Model** | Auto-selected by ROC-AUC score |
| **Saved Files** | `model.joblib`, `scaler.joblib` |

---

## 🩺 Input Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `age` | Age in years |
| 2 | `sex` | 1 = Male, 0 = Female |
| 3 | `cp` | Chest pain type (0–3) |
| 4 | `trestbps` | Resting blood pressure (mm Hg) |
| 5 | `chol` | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| 7 | `restecg` | Resting ECG results (0–2) |
| 8 | `thalach` | Max heart rate achieved |
| 9 | `exang` | Exercise-induced angina (1 = Yes) |
| 10 | `oldpeak` | ST depression induced by exercise |
| 11 | `slope` | Slope of peak exercise ST segment (0–2) |
| 12 | `ca` | Number of major vessels (0–3) |
| 13 | `thal` | Thalassemia (3 = Normal, 6 = Fixed, 7 = Reversible) |

---

## 📊 Output

- **Verdict** — `HIGH RISK` (red) or `LOW RISK` (green)
- **Risk Score** — probability percentage (e.g. `73.4%`) from `predict_proba()`
- **Progress Bar** — visual indicator of risk level

---

## ⚠️ Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `scaler.joblib not found` | `model.py` hasn't been run | Run `python model.py` first |
| `model.joblib not found` | Same as above | Run `python model.py` first |
| `ValueError` on prediction | Non-numeric input in form | Ensure all 13 fields have valid numbers |
| Files not found after training | Wrong working directory | Run both scripts from the same folder |

---

## 🛠️ Tech Stack

- **Backend** — Python 3, Flask
- **ML** — Scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler)
- **Data** — Pandas, NumPy
- **Visualisation** — Matplotlib, Seaborn
- **Serialisation** — Joblib
- **Frontend** — HTML5, CSS3 (Jinja2 templating)

---

## 🚀 Recent Improvements

The following enhancements have been made to improve the performance, reliability, and usability of the system:

- **✅ Fixed Preprocessing Issue**  
  Earlier, the model was trained on scaled data but received raw input during prediction.  
  Now, the same `StandardScaler` is applied during both training and prediction, ensuring consistency and better accuracy.

- **📊 Improved Prediction Accuracy**  
  Model performance has been enhanced by proper data preprocessing and model comparison.

- **🤖 Multiple Model Implementation**  
  Added **Logistic Regression** and **Random Forest** models.  
  The best model is automatically selected based on **ROC-AUC score**.

- **📈 Probability-Based Output**  
  Instead of only showing HIGH/LOW risk, the system now displays **risk percentage** using `predict_proba()`.

- **🎨 UI Enhancement**  
  Improved frontend with:
  - Color-coded risk display (Red = High, Green = Low)
  - Dynamic progress bar for better visualization
  - Cleaner and more user-friendly interface

- **💾 Complete Pipeline Saving**  
  Both `model.joblib` and `scaler.joblib` are saved to ensure consistent predictions even after deployment.

---

## 📌 Disclaimer

> This tool is for **informational and educational purposes only**.  
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
> Always consult a qualified medical professional for health concerns.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.