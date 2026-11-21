import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

# Load all simulation trials
data = pd.read_csv("datasets/noise_robustness_data.csv")

fs = 100
tolerance = 10

sos = butter(
    3,
    [5, 15],
    btype="bandpass",
    fs=fs,
    output="sos"
)

results = []

# Process each noise level and trial independently
for (noise_std, trial), group in data.groupby(
    ["noise_std", "trial"]
):

    group = group.reset_index(drop=True)

    ecg = group["ecg_mV"].to_numpy()
    reported_hr = group["heart_rate_bpm"].to_numpy()
    true_hr = group["true_hr_bpm"].to_numpy()
    actual = group["fault"].to_numpy().astype(bool)

    # Filter ECG
    filtered_ecg = sosfiltfilt(sos, ecg)

    # Detect R-peaks
    peaks, _ = find_peaks(
        filtered_ecg,
        distance=int(0.55 * fs),
        prominence=0.15
    )

    peak_times = group["time_s"].to_numpy()[peaks]

    # Estimate local HR
    estimates = np.full(len(group), np.nan)

    estimation_errors = []
    failed_windows = 0

    for start in range(0, 60, 10):
        end = start + 10

        local_peaks = peak_times[
            (peak_times >= start) &
            (peak_times < end)
        ]

        mask = (
            (group["time_s"] >= start) &
            (group["time_s"] < end)
        )

        indices = np.where(mask)[0]

        if len(local_peaks) >= 2:
            rr = np.diff(local_peaks)
            estimated_hr = 60 / np.median(rr)

            estimates[indices] = estimated_hr

            reference_hr = np.mean(true_hr[indices])

            estimation_errors.append(
                abs(estimated_hr - reference_hr)
            )

        else:
            failed_windows += 1

    # Baseline range detector
    baseline = (
        (reported_hr < 30) |
        (reported_hr > 200)
    )

    # Proposed ECG-HR consistency detector
    proposed = (
        np.isfinite(estimates) &
        (np.abs(reported_hr - estimates) > tolerance)
    )

    # Confusion matrix calculation
    def calculate_metrics(predicted):

        TP = int(np.sum(actual & predicted))
        FP = int(np.sum(~actual & predicted))
        FN = int(np.sum(actual & ~predicted))
        TN = int(np.sum(~actual & ~predicted))

        precision = (
            TP / (TP + FP)
            if TP + FP else 0
        )

        recall = (
            TP / (TP + FN)
            if TP + FN else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        fpr = (
            FP / (FP + TN)
            if FP + TN else 0
        )

        return TP, FP, FN, TN, precision, recall, f1, fpr

    baseline_metrics = calculate_metrics(baseline)
    proposed_metrics = calculate_metrics(proposed)

    mae = (
        np.mean(estimation_errors)
        if estimation_errors else np.nan
    )

    results.append({
        "noise_std": noise_std,
        "trial": trial,
        "r_peaks": len(peaks),
        "failed_windows": failed_windows,
        "hr_mae": mae,
        "baseline_TP": baseline_metrics[0],
        "baseline_FP": baseline_metrics[1],
        "baseline_FN": baseline_metrics[2],
        "baseline_TN": baseline_metrics[3],
        "baseline_precision": baseline_metrics[4],
        "baseline_recall": baseline_metrics[5],
        "baseline_f1": baseline_metrics[6],
        "proposed_TP": proposed_metrics[0],
        "proposed_FP": proposed_metrics[1],
        "proposed_FN": proposed_metrics[2],
        "proposed_TN": proposed_metrics[3],
        "proposed_precision": proposed_metrics[4],
        "proposed_recall": proposed_metrics[5],
        "proposed_f1": proposed_metrics[6],
        "proposed_fpr": proposed_metrics[7]
    })

# Save individual trial results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/noise_robustness_trials.csv",
    index=False
)

# Calculate mean and standard deviation for each noise level
metrics = [
    "r_peaks",
    "failed_windows",
    "hr_mae",
    "baseline_f1",
    "proposed_precision",
    "proposed_recall",
    "proposed_f1",
    "proposed_fpr"
]

summary = results_df.groupby("noise_std")[metrics].agg(
    ["mean", "std"]
)

print("\nNOISE ROBUSTNESS SUMMARY")
print(summary.round(4).to_string())

summary.to_csv(
    "results/noise_robustness_summary.csv"
)

print("\nIndividual trial results saved.")
print("Summary results saved.")
