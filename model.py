# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# Set up plot style
sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# === Load dataset ===
data = pd.read_csv("heart_cleveland_upload.csv")
data.head()

# Basic info
print("Dataset shape:", data.shape)
print("\nMissing values per column:\n", data.isnull().sum())

# Quick summary statistics
data.describe()

# Check column names
data.columns
# Rename the target column to 'target' for consistency
data = data.rename(columns={'condition': 'target'})

# Target variable distribution
sns.countplot(data=data, x='target', palette='Set2')
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler immediately after fit_transform
joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved as scaler.joblib")

print("Number of features:", X_train.shape[1])

# ── ① Logistic Regression ──────────────────────────────────────────────────
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr   = log_reg.predict(X_test_scaled)
y_prob_lr   = log_reg.predict_proba(X_test_scaled)[:, 1]
acc_lr      = accuracy_score(y_test, y_pred_lr)
roc_lr      = roc_auc_score(y_test, y_prob_lr)

print("\n🔹 Logistic Regression Results")
print("Accuracy:", acc_lr)
print("ROC-AUC :", roc_lr)
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# ── ② Random Forest ────────────────────────────────────────────────────────
# Random Forest is an ensemble of decision trees.
# It does NOT require scaled input, but using scaled data here keeps the
# pipeline consistent since the same scaler is shared with app.py.
rf_model = RandomForestClassifier(
    n_estimators=100,   # 100 trees — good balance of speed vs accuracy
    random_state=42,    # reproducibility
    max_depth=None,     # grow full trees; tune if overfitting is observed
    n_jobs=-1           # use all available CPU cores
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf   = rf_model.predict(X_test_scaled)
y_prob_rf   = rf_model.predict_proba(X_test_scaled)[:, 1]
acc_rf      = accuracy_score(y_test, y_pred_rf)
roc_rf      = roc_auc_score(y_test, y_prob_rf)

print("\n🔸 Random Forest Results")
print("Accuracy:", acc_rf)
print("ROC-AUC :", roc_rf)
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# ── ③ Side-by-side accuracy comparison ────────────────────────────────────
print("\n" + "=" * 45)
print("       MODEL COMPARISON SUMMARY")
print("=" * 45)
print(f"  {'Model':<25} {'Accuracy':>8}  {'ROC-AUC':>8}")
print("-" * 45)
print(f"  {'Logistic Regression':<25} {acc_lr:>8.4f}  {roc_lr:>8.4f}")
print(f"  {'Random Forest':<25} {acc_rf:>8.4f}  {roc_rf:>8.4f}")
print("=" * 45)

# Bar chart comparison
fig, ax = plt.subplots(figsize=(7, 4))
models  = ['Logistic Regression', 'Random Forest']
acc_scores = [acc_lr, acc_rf]
roc_scores = [roc_lr, roc_rf]
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, acc_scores, width, label='Accuracy',  color='#7A57E8')
bars2 = ax.bar(x + width/2, roc_scores, width, label='ROC-AUC',   color='#B39DDB')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Logistic Regression vs Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', padding=3)
ax.bar_label(bars2, fmt='%.3f', padding=3)
plt.tight_layout()
plt.show()

# ── ④ ROC Curves — both models on the same plot ────────────────────────────
plt.figure(figsize=(6, 5))
RocCurveDisplay.from_estimator(log_reg,  X_test_scaled, y_test, name="Logistic Regression", ax=plt.gca())
RocCurveDisplay.from_estimator(rf_model, X_test_scaled, y_test, name="Random Forest",        ax=plt.gca())
plt.title("ROC Curves Comparison")
plt.show()

# ── ⑤ Random Forest — feature importance plot ─────────────────────────────
feature_names = X.columns.tolist()
importances   = rf_model.feature_importances_
sorted_idx    = np.argsort(importances)

plt.figure(figsize=(7, 5))
plt.barh(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx],
    color='#5D3FD3'
)
plt.xlabel("Importance Score")
plt.title("Random Forest — Feature Importances")
plt.tight_layout()
plt.show()

# ── ⑥ Save the better-performing model ────────────────────────────────────
# Automatically pick the model with the higher ROC-AUC score.
if roc_rf >= roc_lr:
    best_model      = rf_model
    best_model_name = "Random Forest"
else:
    best_model      = log_reg
    best_model_name = "Logistic Regression"

joblib.dump(best_model, 'model.joblib')
print(f"\nBest model ({best_model_name}) saved as model.joblib")

print("\n✅ Both files saved successfully:")
print("   • model.joblib  — best model:", best_model_name)
print("   • scaler.joblib — fitted StandardScaler (mean & std from training data)")