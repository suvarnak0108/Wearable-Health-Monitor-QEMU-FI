import pandas as pd
import numpy as np

np.random.seed(42)

# Load variable noisy ECG dataset
data = pd.read_csv("datasets/variable_noisy_ecg.csv")

# Create reported HR from simulated sensor HR
data["heart_rate_bpm"] = data["true_hr_bpm"].copy()

# Select 150 random samples
N = len(data)
num_faults = 150

fault_indices = np.random.choice(
    N,
    num_faults,
    replace=False
)

# Inject subtle in-range HR faults
data.loc[fault_indices, "heart_rate_bpm"] = 110

# Save ground truth
ground_truth = pd.DataFrame({
    "sample_index": fault_indices
})

ground_truth.to_csv(
    "datasets/variable_hr_ground_truth.csv",
    index=False
)

# Save corrupted dataset
data.to_csv(
    "datasets/variable_hr_fault_data.csv",
    index=False
)

print("Variable HR fault injection completed.")
print("Total samples:", N)
print("Injected faults:", num_faults)
print("\nReported HR distribution:")
print(data["heart_rate_bpm"].value_counts().sort_index())
