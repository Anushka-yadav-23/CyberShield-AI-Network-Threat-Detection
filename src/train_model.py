
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "network_traffic.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. Basic data cleaning
df = df.drop_duplicates().dropna().reset_index(drop=True)

# 3. Encode text features and labels
df["protocol"] = df["protocol"].astype(str)
protocol_encoder = LabelEncoder()
df["protocol"] = protocol_encoder.fit_transform(df["protocol"])

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# 4. Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 6. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 7. Evaluate on the held-out test split
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    labels=list(range(len(label_encoder.classes_))),
    target_names=label_encoder.classes_,
    zero_division=0,
)
matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(label_encoder.classes_))),
)

# 8. Save model and encoders together
joblib.dump(
    {
        "model": model,
        "protocol_encoder": protocol_encoder,
        "label_encoder": label_encoder,
        "features": list(X.columns),
    },
    MODEL_DIR / "threat_model.joblib",
)

# 9. Save evaluation results
metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "classes": label_encoder.classes_.tolist(),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
}

with open(OUTPUT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 10. Save predictions for dashboard use
predictions = X_test.copy()
predictions["actual"] = label_encoder.inverse_transform(y_test)
predictions["predicted"] = label_encoder.inverse_transform(y_pred)
predictions["correct"] = predictions["actual"] == predictions["predicted"]
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

print("=" * 55)
print("AI CYBERSECURITY THREAT DETECTION - MODEL EVALUATION")
print("=" * 55)
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"Accuracy:         {accuracy:.4f}")
print("\nClassification report:")
print(report)
print("\nConfusion matrix:")
print(matrix)
print("\nSaved model: models/threat_model.joblib")
print("Saved metrics: outputs/evaluation.json")
print("Saved predictions: outputs/predictions.csv")
print("\nNOTE: Metrics are from synthetic simulated data.")
