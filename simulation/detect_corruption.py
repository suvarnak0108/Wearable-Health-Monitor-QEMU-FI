
import pandas as pd

# Load corrupted dataset and ground truth
data = pd.read_csv("datasets/corrupted_data.csv")
truth = pd.read_csv("datasets/corruption_ground_truth.csv")

# Identify invalid measurements
detected_indices = set()

for index, row in data.iterrows():

    hr_invalid = not (30 <= row["heart_rate_bpm"] <= 200)
    spo2_invalid = not (70 <= row["spo2_pct"] <= 100)

    if hr_invalid or spo2_invalid:
        detected_indices.add(index)

# Actual corrupted sample indices
actual_indices = set(truth["sample_index"].astype(int))

# Calculate confusion matrix
TP = len(detected_indices & actual_indices)
FP = len(detected_indices - actual_indices)
FN = len(actual_indices - detected_indices)
TN = len(data) - TP - FP - FN

# Calculate performance metrics
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\n--- Corrupted Measurement Detection ---")
print("Total samples:", len(data))
print("Actual corrupted:", len(actual_indices))
print("Detected corrupted:", len(detected_indices))

print("\nTrue positives:", TP)
print("False positives:", FP)
print("False negatives:", FN)
print("True negatives:", TN)

print("\nPrecision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))

# Save detected indices
pd.DataFrame({
    "detected_index": sorted(detected_indices)
}).to_csv("results/detected_corruption.csv", index=False)

# Save metrics
pd.DataFrame([{
    "TP": TP,
    "FP": FP,
    "FN": FN,
    "TN": TN,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_score": f1 * 100
}]).to_csv("results/corruption_metrics.csv", index=False)
