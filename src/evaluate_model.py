
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "threat_model.joblib"
DATA_PATH = ROOT / "data" / "network_traffic.csv"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "images"

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

# Load the trained model and its encoders
saved = joblib.load(MODEL_PATH)
model = saved["model"]
protocol_encoder = saved["protocol_encoder"]
label_encoder = saved["label_encoder"]
features = saved["features"]

# Load and prepare network traffic
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna().reset_index(drop=True)
df["protocol"] = protocol_encoder.transform(df["protocol"])

X = df[features]
actual = df["label"].astype(str)

# Predict traffic categories
predicted_ids = model.predict(X)
predicted = label_encoder.inverse_transform(predicted_ids)

results = df.copy()
results["actual_label"] = actual
results["predicted_label"] = predicted
results["alert"] = results["predicted_label"] != "Normal"
results["status"] = results["predicted_label"].apply(
    lambda label: "SUSPICIOUS" if label != "Normal" else "NORMAL"
)

# Save the simulated detection report
results.to_csv(OUTPUT_DIR / "threat_alerts.csv", index=False)

# Threat distribution chart
plt.figure(figsize=(9, 5))
sns.countplot(
    data=results,
    x="predicted_label",
    order=label_encoder.classes_,
)
plt.title("AI-Predicted Network Traffic Categories")
plt.xlabel("Predicted category")
plt.ylabel("Number of records")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "threat_distribution.png", dpi=150)
plt.close()

# Confusion matrix for the complete synthetic dataset
matrix = confusion_matrix(
    actual,
    predicted,
    labels=label_encoder.classes_,
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.title("Confusion Matrix — Synthetic Data")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
plt.close()

summary = results["predicted_label"].value_counts().to_dict()
alerts = results[results["alert"]]

report = {
    "total_records": int(len(results)),
    "predicted_categories": {
        str(key): int(value) for key, value in summary.items()
    },
    "predicted_suspicious_records": int(len(alerts)),
    "note": (
        "Synthetic demonstration. These are model predictions, "
        "not verified real-world security incidents."
    ),
}

with open(OUTPUT_DIR / "alert_summary.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("=" * 45)
print("THREAT DETECTION SUMMARY")
print("=" * 45)
print(f"Traffic records analysed: {len(results)}")
print(f"Predicted suspicious records: {len(alerts)}")
print("\nPredicted categories:")
print(results["predicted_label"].value_counts())
print("\nFiles created:")
print("outputs/threat_alerts.csv")
print("outputs/alert_summary.json")
print("images/threat_distribution.png")
print("images/confusion_matrix.png")
print("\nSynthetic demo only — not production security monitoring.")
