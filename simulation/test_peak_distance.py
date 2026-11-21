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

    for distance_sec in [0.45, 0.55, 0.60]:

        peaks, _ = find_peaks(
            filtered,
            distance=int(distance_sec * fs),
            prominence=0.15
        )

        peak_times = group["time_s"].to_numpy()[peaks]

        print(
            f"\nTrial {trial}, "
            f"minimum distance={distance_sec}s"
        )

        for start in range(0, 60, 10):

            local = peak_times[
                (peak_times >= start) &
                (peak_times < start + 10)
            ]

            if len(local) >= 2:
                rr = np.diff(local)
                hr = 60 / np.median(rr)
            else:
                hr = np.nan

            print(
                f"{start}-{start+10}s: "
                f"Peaks={len(local)}, "
                f"Estimated HR={hr:.2f}"
            )
