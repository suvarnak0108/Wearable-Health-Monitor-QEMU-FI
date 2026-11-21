
import pandas as pd

# Load normal dataset
data = pd.read_csv("datasets/normal_data.csv")

# Configuration
sampling_rate = 100
outage_duration = 0.5
samples_per_outage = int(sampling_rate * outage_duration)

# Three outage starting positions
outage_starts = [500, 1500, 2500]

missing_indices = []

# Identify samples lost during interruptions
for start in outage_starts:
    missing_indices.extend(
        range(start, start + samples_per_outage)
    )

# Save ground truth
pd.DataFrame({
    "sample_index": missing_indices
}).to_csv(
    "datasets/communication_ground_truth.csv",
    index=False
)

# Remove unavailable samples
faulty_data = data.drop(missing_indices).reset_index(drop=True)

# Save communication-fault dataset
faulty_data.to_csv(
    "datasets/communication_data.csv",
    index=False
)

print("Communication fault injection completed!")
print("Original samples:", len(data))
print("Number of interruptions:", len(outage_starts))
print("Duration per interruption:", outage_duration, "seconds")
print("Missing samples:", len(missing_indices))
print("Remaining samples:", len(faulty_data))
