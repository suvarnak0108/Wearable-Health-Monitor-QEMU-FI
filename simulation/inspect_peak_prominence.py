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

    peaks, properties = find_peaks(
        filtered,
        distance=int(0.45 * fs),
        prominence=0.15
    )

    times = group["time_s"].to_numpy()[peaks]
    prominences = properties["prominences"]

    print(f"\nTRIAL {trial}")

    for t, p in zip(times, prominences):
        if t < 10:
            print(
                f"Time: {t:.2f}s | "
                f"Prominence: {p:.3f}"
            )
