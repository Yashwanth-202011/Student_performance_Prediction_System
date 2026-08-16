"""
train_model.py
----------------
Trains a machine learning model to predict whether a student will
PASS or FAIL based on study habits, attendance and past academic
performance.

Dataset : UCI "Student Performance" Data Set (student-mat.csv)
Source  : Cortez, P. & Silva, A. (2008), UCI Machine Learning Repository
          https://archive.ics.uci.edu/dataset/320/student+performance
Records : 395 students, Mathematics course, Portuguese secondary schools

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATASET_PATH = os.path.join("dataset", "student-mat.csv")
MODEL_PATH = os.path.join("model", "trained_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

# -----------------------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------------------
print("Step 1: Loading dataset ...")
df = pd.read_csv(DATASET_PATH, sep=";")
print(f"  -> Loaded {df.shape[0]} records with {df.shape[1]} columns.")

# -----------------------------------------------------------------
# 2. Clean the data / handle missing values
# -----------------------------------------------------------------
print("Step 2: Cleaning data ...")
missing = df.isnull().sum().sum()
print(f"  -> Missing values found: {missing}")
if missing > 0:
    df = df.fillna(df.median(numeric_only=True))
    print("  -> Missing numeric values filled with column median.")

# -----------------------------------------------------------------
# 3. Feature selection
#    We use features a student can realistically self-report:
#      - studytime : weekly study hours (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
#      - absences  : number of school absences  (attendance proxy)
#      - failures  : number of past class failures (academic history)
#      - G1        : first period grade  (0-20)  -> "previous marks"
#      - G2        : second period grade (0-20)  -> "internal marks"
#    Target:
#      - pass  : 1 if final grade G3 >= 10 (out of 20), else 0
# -----------------------------------------------------------------
print("Step 3: Selecting features and target ...")
FEATURES = ["studytime", "absences", "failures", "G1", "G2"]
df["pass"] = (df["G3"] >= 10).astype(int)

X = df[FEATURES]
y = df["pass"]
print(f"  -> Features: {FEATURES}")
print(f"  -> Target: pass (1) / fail (0)")
print(f"  -> Class balance: {y.value_counts().to_dict()}")

# -----------------------------------------------------------------
# 4. Preprocessing - scale numeric features
# -----------------------------------------------------------------
print("Step 4: Scaling features ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------
# 5. Train / test split
# -----------------------------------------------------------------
print("Step 5: Splitting into train and test sets (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  -> Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------------------------------------------------
# 6. Train the model
#    Random Forest is chosen because:
#      - It handles the mix of small-scale numeric features well
#      - It is robust to overfitting compared to a single Decision Tree
#      - It requires little tuning, making it easy to explain in a viva
#      - It provides feature importance for interpretability
# -----------------------------------------------------------------
print("Step 6: Training Random Forest Classifier ...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# 7. Evaluate the model
# -----------------------------------------------------------------
print("Step 7: Evaluating model on test data ...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n===== Evaluation Metrics =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

print("\nFeature Importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:12s}: {imp:.4f}")

# -----------------------------------------------------------------
# 8. Save the trained model and scaler locally
# -----------------------------------------------------------------
print("\nStep 8: Saving trained model and scaler ...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"  -> Model saved to  {MODEL_PATH}")
print(f"  -> Scaler saved to {SCALER_PATH}")

# Save metrics to a small text file so app.py / README can reference them
with open(os.path.join("model", "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print("\nTraining complete.")
