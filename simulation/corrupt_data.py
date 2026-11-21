
import pandas as pd
import numpy as np

# Load original dataset
data = pd.read_csv("datasets/normal_data.csv")

# Configuration
fault_percentage = 5
total_samples = len(data)
corrupt_count = int(total_samples * fault_percentage / 100)

# Reproducible random selection
np.random.seed(42)

corrupt_indices = np.random.choice(
    data.index,
    size=corrupt_count,
    replace=False
)

# Divide corrupted samples between two channels
half = corrupt_count // 2

hr_indices = corrupt_indices[:half]
spo2_indices = corrupt_indices[half:]

# Inject invalid values
data.loc[hr_indices, "heart_rate_bpm"] = 250
data.loc[spo2_indices, "spo2_pct"] = 105

# Save ground truth
ground_truth = pd.DataFrame({
    "sample_index": corrupt_indices,
    "corrupted_channel": [
        "heart_rate" if i in hr_indices else "spo2"
        for i in corrupt_indices
    ]
})

ground_truth.to_csv(
    "datasets/corruption_ground_truth.csv",
    index=False
)

# Save corrupted dataset
data.to_csv(
    "datasets/corrupted_data.csv",
    index=False
)

print("Corruption injection completed!")
print("Original samples:", total_samples)
print("Corrupted samples:", corrupt_count)
print("HR corrupted:", len(hr_indices))
print("SpO2 corrupted:", len(spo2_indices))
print("Remaining normal samples:", total_samples-corrupt_count)
