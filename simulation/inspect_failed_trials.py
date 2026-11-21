import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

data = pd.read_csv("datasets/noise_robustness_data.csv")

fs = 100

sos = butter(
    3, [5, 15],
    btype="bandpass",
    fs=fs,
    output="sos"
)

for trial in [1, 5]:

    group = data[
        (data["noise_std"] == 0.35) &
        (data["trial"] == trial)
    ].reset_index(drop=True)

    ecg = group["ecg_mV"].to_numpy()
    filtered = sosfiltfilt(sos, ecg)

    peaks, _ = find_peaks(
        filtered,
        distance=int(0.45 * fs),
        prominence=0.15
    )

    peak_times = group["time_s"].to_numpy()[peaks]

    print(f"\nTRIAL {trial}")

    for start in range(0, 60, 10):

        end = start + 10

        local_peaks = peak_times[
            (peak_times >= start) &
            (peak_times < end)
        ]

        if len(local_peaks) >= 2:
            rr = np.diff(local_peaks)
            estimated_hr = 60 / np.median(rr)
        else:
            estimated_hr = np.nan

        true_hr = group.loc[
            (group["time_s"] >= start) &
            (group["time_s"] < end),
            "true_hr_bpm"
        ].iloc[0]

        print(
            f"{start}-{end}s | "
            f"True HR: {true_hr:.1f} | "
            f"Estimated HR: {estimated_hr:.2f} | "
            f"Peaks: {len(local_peaks)}"
        )
