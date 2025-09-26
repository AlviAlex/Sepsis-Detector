import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import numpy as np

# 📂 Paths to training folders
folders = [
    r"C:\Users\Alvin\Downloads\training_setA\training",
    r"C:\Users\Alvin\Downloads\training_setB\training_setB"
]

# Load all .psv files
all_data = []
for folder_path in folders:
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue
    psv_files = [f for f in os.listdir(folder_path) if f.endswith('.psv')]
    for file in psv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, sep='|')
        all_data.append(df)
    print(f"Loaded {len(psv_files)} files from {folder_path}")

# Combine everything
df = pd.concat(all_data, ignore_index=True)
print(f"\n✅ Total dataset shape: {df.shape}")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# 🔑 Use all features except ID + label
features = [col for col in df.columns if col not in ["SepsisLabel", "Patient_ID"]]
X = df[features]
y = df["SepsisLabel"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ⚖️ SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {sum(y_train_res)} sepsis / {len(y_train_res) - sum(y_train_res)} healthy")

# Train XGBoost
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# 🔥 Test different thresholds
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    print(f"\n=== Threshold = {t} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "sepsis_xgb_balanced.pkl")
print("\n✅ Balanced model saved as sepsis_xgb_balanced.pkl")
