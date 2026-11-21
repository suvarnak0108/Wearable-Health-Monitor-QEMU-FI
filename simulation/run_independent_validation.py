import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

data = pd.read_csv(
    "datasets/independent_validation_data.csv"
)

fs = 100
tolerance = 10

sos = butter(
    3, [5, 15],
    btype="bandpass",
    fs=fs,
    output="sos"
)

results = []

for (noise_std, trial), group in data.groupby(
    ["noise_std", "trial"]
):

    group = group.reset_index(drop=True)

    ecg = group["ecg_mV"].to_numpy()
    reported_hr = group["heart_rate_bpm"].to_numpy()
    actual = group["fault"].to_numpy().astype(bool)

    # ECG filtering
    filtered = sosfiltfilt(sos, ecg)

    # Improved R-peak detection
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.55 * fs),
        prominence=0.15
    )

    peak_times = group["time_s"].to_numpy()[peaks]

    estimates = np.full(len(group), np.nan)
    errors = []
    failed_windows = 0

    # Local HR estimation
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

            reference_hr = np.mean(
                group["true_hr_bpm"].to_numpy()[indices]
            )

            errors.append(
                abs(estimated_hr - reference_hr)
            )

        else:
            failed_windows += 1

    # Baseline range detector
    baseline = (
        (reported_hr < 30) |
        (reported_hr > 200)
    )

    # Proposed consistency detector
    proposed = (
        np.isfinite(estimates) &
        (np.abs(reported_hr - estimates) > tolerance)
    )

    def evaluate(predicted):

        TP = int(np.sum(actual & predicted))
        FP = int(np.sum(~actual & predicted))
        FN = int(np.sum(actual & ~predicted))
        TN = int(np.sum(~actual & ~predicted))

        precision = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        fpr = FP / (FP + TN) if FP + TN else 0

        return TP, FP, FN, TN, precision, recall, f1, fpr

    b = evaluate(baseline)
    p = evaluate(proposed)

    results.append({
        "noise_std": noise_std,
        "trial": trial,
        "r_peaks": len(peaks),
        "failed_windows": failed_windows,
        "hr_mae": np.mean(errors) if errors else np.nan,
        "baseline_f1": b[6],
        "proposed_TP": p[0],
        "proposed_FP": p[1],
        "proposed_FN": p[2],
        "proposed_TN": p[3],
        "proposed_precision": p[4],
        "proposed_recall": p[5],
        "proposed_f1": p[6],
        "proposed_fpr": p[7]
    })

# Save individual trial results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/independent_validation_trials.csv",
    index=False
)

# Summary
metrics = [
    "r_peaks",
    "failed_windows",
    "hr_mae",
    "proposed_precision",
    "proposed_recall",
    "proposed_f1",
    "proposed_fpr"
]

summary = results_df.groupby("noise_std")[metrics].agg(
    ["mean", "std"]
)

summary.to_csv(
    "results/independent_validation_summary.csv"
)

print("\nINDEPENDENT VALIDATION RESULTS")
print(summary.round(4).to_string())

print("\nTotal trials:", len(results_df))
print("Results saved successfully.")
