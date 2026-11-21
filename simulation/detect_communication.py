
import pandas as pd
import numpy as np

# Load dataset and ground truth
data = pd.read_csv("datasets/communication_data.csv")
truth = pd.read_csv("datasets/communication_ground_truth.csv")

sampling_rate = 100
expected_interval = 1 / sampling_rate

# Convert timestamps into sample indices
received = set(
    np.rint(data["time_s"].to_numpy() * sampling_rate).astype(int)
)

actual = set(
    truth["sample_index"].astype(int)
)

# Detect missing samples using timestamp gaps
timestamps = data["time_s"].to_numpy()
detected = set()

for i in range(1, len(timestamps)):
    gap = round(
        (timestamps[i] - timestamps[i-1]) / expected_interval
    ) - 1

    if gap > 0:
        previous = round(timestamps[i-1] * sampling_rate)

        for j in range(1, gap + 1):
            detected.add(previous + j)

# Calculate metrics
TP = len(detected & actual)
FP = len(detected - actual)
FN = len(actual - detected)
TN = len(data) + len(actual) - TP - FP - FN

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

# Count separate interruption events
events = 0
previous_index = None

for index in sorted(detected):
    if previous_index is None or index != previous_index + 1:
        events += 1
    previous_index = index

print("\n--- Communication Interruption Results ---")
print("Actual missing samples:", len(actual))
print("Detected missing samples:", len(detected))
print("Detected interruption events:", events)

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
    "Interruption_events": events,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1_score": f1 * 100
}]).to_csv(
    "results/communication_metrics.csv",
    index=False
)
