# =============================================================================
# ❤️ HEART DISEASE PREDICTION SYSTEM
# =============================================================================
# This project:
# → Loads the heart disease dataset (kaggle dataset)
# → Trains Machine Learning models
# → Compares model performance
# → Saves the best model for future prediction
# =============================================================================



# =============================================================================
# 📌 IMPORT REQUIRED LIBRARIES
# =============================================================================

# Used for handling dataset
import pandas as pd
import numpy as np

# Used for graphs and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Used for saving trained model and scaler
import joblib

# Used for splitting dataset into train and test data
from sklearn.model_selection import train_test_split

# Used for scaling/standardizing data
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Used for checking model performance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)



# =============================================================================
# 📌 SET GRAPH STYLE
# =============================================================================
# This makes graphs look cleaner and more attractive

sns.set(style="whitegrid", palette="pastel")



# =============================================================================
# 📌 LOAD DATASET
# =============================================================================
# Reading CSV file

data = pd.read_csv("heart_cleveland_upload.csv")


# Rename target column for easier understanding
# condition → target

data = data.rename(columns={'condition': 'target'})



# =============================================================================
# 📌 DATASET INFORMATION
# =============================================================================

# Print total rows and columns
print("Dataset Shape:", data.shape)

# Check if dataset has any missing/null values
print("\nMissing Values:\n")

print(data.isnull().sum())



# =============================================================================
# 📌 TARGET DISTRIBUTION GRAPH
# =============================================================================
# Shows how many patients:
# → Have heart disease
# → Do not have heart disease

plt.figure(figsize=(6, 4))

sns.countplot(
    data=data,
    x='target',
    hue='target',
    palette='Set2',
    legend=False
)

plt.title("Target Distribution")

plt.xlabel("0 = No Disease | 1 = Disease")

plt.tight_layout()

# Save graph image
plt.savefig("target_distribution.png")



# =============================================================================
# 📌 CORRELATION HEATMAP
# =============================================================================
# Shows relationship between all features which help in Analysis
# Darker colors = stronger relationship

plt.figure(figsize=(10, 8))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

# Save heatmap image
plt.savefig("correlation_heatmap.png")



# =============================================================================
# 📌 SPLIT INPUT AND OUTPUT DATA
# =============================================================================

# X contains input features
X = data.drop("target", axis=1)

# y contains output/target value
y = data["target"]



# =============================================================================
# 📌 TRAIN TEST SPLIT
# =============================================================================
# 80% data → Training
# 20% data → Testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# =============================================================================
# 📌 FEATURE SCALING
# =============================================================================
# This code scales all features into a similar range because different features have different value ranges.”

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# =============================================================================
# 📌 SAVE SCALER
# =============================================================================
# Save scaler file for Flask prediction

joblib.dump(scaler, 'scaler.joblib')

print("\n✅ scaler.joblib saved successfully")



# =============================================================================
# 📌 LOGISTIC REGRESSION MODEL
# =============================================================================
# Create Logistic Regression model

log_reg = LogisticRegression(max_iter=1000)

# Train model using training data
log_reg.fit(X_train_scaled, y_train)

# Predict output using test data
y_pred_lr = log_reg.predict(X_test_scaled)

# Predict probability score
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
acc_lr = accuracy_score(y_test, y_pred_lr)

# Calculate ROC-AUC score
roc_lr = roc_auc_score(y_test, y_prob_lr)



# =============================================================================
# 📌 LOGISTIC REGRESSION RESULTS
# =============================================================================

print("\n================ LOGISTIC REGRESSION ================\n")

print("Accuracy :", acc_lr)

print("ROC-AUC  :", roc_lr)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_lr))



# =============================================================================
# 📌 RANDOM FOREST MODEL
# =============================================================================
# Create Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train_scaled, y_train)

# Predict output
y_pred_rf = rf_model.predict(X_test_scaled)

# Predict probability score
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
acc_rf = accuracy_score(y_test, y_pred_rf)

# Calculate ROC-AUC score
roc_rf = roc_auc_score(y_test, y_prob_rf)



# =============================================================================
# 📌 RANDOM FOREST RESULTS
# =============================================================================

print("\n================ RANDOM FOREST ================\n")

print("Accuracy :", acc_rf)

print("ROC-AUC  :", roc_rf)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred_rf))



# =============================================================================
# 📌 MODEL COMPARISON
# =============================================================================
# Compare both models side by side

print("\n================ MODEL COMPARISON ================\n")

print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")

print("-" * 50)

print(f"{'Logistic Regression':<25} {acc_lr:>10.4f} {roc_lr:>10.4f}")

print(f"{'Random Forest':<25} {acc_rf:>10.4f} {roc_rf:>10.4f}")



# =============================================================================
# 📌 PERFORMANCE COMPARISON GRAPH
# =============================================================================
# Create bar graph for model comparison

fig, ax = plt.subplots(figsize=(7, 4))

# Model names
models = ['Logistic Regression', 'Random Forest']

# Accuracy values
acc_scores = [acc_lr, acc_rf]

# ROC-AUC values
roc_scores = [roc_lr, roc_rf]

# Position setup for bars
x = np.arange(len(models))

width = 0.35

# Accuracy bars
bars1 = ax.bar(
    x - width / 2,
    acc_scores,
    width,
    label='Accuracy'
)

# ROC-AUC bars
bars2 = ax.bar(
    x + width / 2,
    roc_scores,
    width,
    label='ROC-AUC'
)

# Labels and title
ax.set_ylabel('Score')

ax.set_title('Model Comparison')

ax.set_xticks(x)

ax.set_xticklabels(models)

ax.legend()

# Show values on bars
ax.bar_label(bars1, fmt='%.3f')

ax.bar_label(bars2, fmt='%.3f')

plt.tight_layout()

# Save graph
plt.savefig("model_comparison.png")



# =============================================================================
# 📌 ROC CURVE GRAPH
# =============================================================================
# ROC curve helps compare classification performance

plt.figure(figsize=(6, 5))

# Logistic Regression ROC Curve
RocCurveDisplay.from_estimator(
    log_reg,
    X_test_scaled,
    y_test,
    name="Logistic Regression",
    ax=plt.gca()
)

# Random Forest ROC Curve
RocCurveDisplay.from_estimator(
    rf_model,
    X_test_scaled,
    y_test,
    name="Random Forest",
    ax=plt.gca()
)

plt.title("ROC Curve Comparison")

plt.tight_layout()

# Save graph
plt.savefig("roc_curve.png")



# =============================================================================
# 📌 FEATURE IMPORTANCE GRAPH
# =============================================================================
# Shows which features are most important
# for heart disease prediction

feature_names = X.columns.tolist()

importances = rf_model.feature_importances_

sorted_idx = np.argsort(importances)

plt.figure(figsize=(7, 5))

plt.barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx]
)

plt.xlabel("Importance Score")

plt.title("Feature Importance")

plt.tight_layout()

# Save graph
plt.savefig("feature_importance.png")



# =============================================================================
# 📌 SELECT BEST MODEL
# =============================================================================
# Choose model with higher ROC-AUC score

if roc_rf >= roc_lr:

    best_model = rf_model

    best_model_name = "Random Forest"

else:

    best_model = log_reg

    best_model_name = "Logistic Regression"



# =============================================================================
# 📌 SAVE BEST MODEL
# =============================================================================
# Save best trained model

joblib.dump(best_model, 'model.joblib')

print(f"\n✅ Best Model Saved : {best_model_name}")



# =============================================================================
# 📌 FINAL SUCCESS MESSAGE
# =============================================================================

print("\n==================================================")

print("✅ model.joblib saved")

print("✅ scaler.joblib saved")

print(f"✅ Best Model : {best_model_name}")

print("🎉 Project Completed Successfully!")

print("==================================================")