
import pandas as pd
import numpy as np

# Load original dataset
data = pd.read_csv("datasets/normal_data.csv")

# Configuration
corrupt_percentage = 5
corrupt_count = int(len(data) * corrupt_percentage / 100)

np.random.seed(42)

# Select random sample positions
corrupt_indices = np.random.choice(
    data.index,
    size=corrupt_count,
    replace=False
)

# Introduce incorrect but in-range HR readings
data.loc[corrupt_indices, "heart_rate_bpm"] = 110

# Save ground truth
pd.DataFrame({
    "sample_index": corrupt_indices
}).to_csv(
    "datasets/subtle_ground_truth.csv",
    index=False
)

# Save corrupted dataset
data.to_csv(
    "datasets/subtle_corrupted_data.csv",
    index=False
)

print("Subtle corruption injection completed!")
print("Original samples:", len(data))
print("Corrupted samples:", corrupt_count)
print("Normal HR: 75 BPM")
print("Injected HR: 110 BPM")
print("All corrupted values remain within the configured range.")
