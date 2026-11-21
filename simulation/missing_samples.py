
import pandas as pd
import numpy as np

# Load normal dataset
data = pd.read_csv("datasets/normal_data.csv")

# Simulation configuration
fault_percentage = 5
random_seed = 42

# Calculate number of samples to remove
total_samples = len(data)
missing_count = int(total_samples * fault_percentage / 100)

# Select samples reproducibly
np.random.seed(random_seed)
missing_indices = np.random.choice(
    data.index,
    size=missing_count,
    replace=False
)

# Save ground-truth missing timestamps
missing_times = data.loc[missing_indices, ["time_s"]]
missing_times.to_csv(
    "datasets/missing_ground_truth.csv",
    index=False
)

# Remove selected samples
faulty_data = data.drop(missing_indices).reset_index(drop=True)

# Save fault-injected dataset
faulty_data.to_csv(
    "datasets/missing_data.csv",
    index=False
)

print("Missing sample injection completed!")
print("Original samples:", total_samples)
print("Missing samples:", missing_count)
print("Remaining samples:", len(faulty_data))
print("Fault percentage:", fault_percentage, "%")
