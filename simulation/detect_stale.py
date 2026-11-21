
import pandas as pd

# Load dataset and ground truth
data = pd.read_csv("datasets/stale_data.csv")
truth = pd.read_csv("datasets/stale_ground_truth.csv")

# Detection threshold
threshold = 0.02

# Calculate measurement age
age = data["time_s"] - data["source_time_s"]

# Identify stale samples
detected = set(
    data.index[age > threshold]
)

actual = set(
    truth["sample_index"].astype(int)
)

# Confusion matrix
TP = len(detected & actual)
FP = len(detected - actual)
FN = len(actual - detected)
TN = len(data) - TP - FP - FN

# Performance metrics
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n--- Stale Data Detection Results ---")
print("Total samples:", len(data))
print("Actual stale samples:", len(actual))
print("Detected stale samples:", len(detected))

print("\nTP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

print("\nPrecision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))

# Save results
pd.DataFrame([{
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_score": f1 * 100
}]).to_csv("results/stale_metrics.csv", index=False)
