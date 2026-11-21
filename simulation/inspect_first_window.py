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

    # Inspect only first 10 seconds
    first_window = peak_times[peak_times < 10]

    rr = np.diff(first_window)

    print(f"\nTRIAL {trial}")
    print("Peak times:", np.round(first_window, 3))
    print("RR intervals:", np.round(rr, 3))
    print("Median RR:", round(np.median(rr), 4))
    print("Estimated HR:", round(60 / np.median(rr), 2))

