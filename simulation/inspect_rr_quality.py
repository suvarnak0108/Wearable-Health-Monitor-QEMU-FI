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

    filtered = sosfiltfilt(
        sos, group["ecg_mV"].to_numpy()
    )

    peaks, _ = find_peaks(
        filtered,
        distance=45,
        prominence=0.15
    )

    times = group["time_s"].to_numpy()[peaks]

    print(f"\nTRIAL {trial}")

    for start in range(0, 60, 10):

        local = times[
            (times >= start) &
            (times < start + 10)
        ]

        rr = np.diff(local)

        if len(rr) > 0:
            median_rr = np.median(rr)
            cv = np.std(rr) / np.mean(rr)

            print(
                f"{start}-{start+10}s | "
                f"Median RR: {median_rr:.3f}s | "
                f"RR CV: {cv:.3f} | "
                f"Intervals: {len(rr)}"
            )
