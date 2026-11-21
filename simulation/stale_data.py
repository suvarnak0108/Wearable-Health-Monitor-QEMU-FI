
import pandas as pd
import numpy as np

# Load normal physiological dataset
original = pd.read_csv("datasets/normal_data.csv")

data = original.copy()

# Add source timestamp
data["source_time_s"] = data["time_s"]

# Configuration
fault_percentage = 5
stale_delay_samples = 5
sampling_rate = 100

total = len(data)
stale_count = int(total * fault_percentage / 100)

# Select samples, excluding first five
np.random.seed(42)

stale_indices = np.random.choice(
    np.arange(stale_delay_samples, total),
    size=stale_count,
    replace=False
)

# Inject stale measurements
for index in stale_indices:

    old_index = index - stale_delay_samples

    # Copy old physiological measurements
    data.loc[index, "ecg_mV"] = original.loc[old_index, "ecg_mV"]
    data.loc[index, "heart_rate_bpm"] = original.loc[old_index, "heart_rate_bpm"]
    data.loc[index, "spo2_pct"] = original.loc[old_index, "spo2_pct"]

    # Retain original acquisition timestamp
    data.loc[index, "source_time_s"] = original.loc[old_index, "time_s"]

# Save ground truth
pd.DataFrame({
    "sample_index": stale_indices
}).to_csv(
    "datasets/stale_ground_truth.csv",
    index=False
)

# Save stale dataset
data.to_csv(
    "datasets/stale_data.csv",
    index=False
)

print("Stale data injection completed!")
print("Original samples:", total)
print("Stale samples:", stale_count)
print("Remaining normal samples:", total - stale_count)
print("Stale duration: 50 ms")
