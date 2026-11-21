import pandas as pd
import numpy as np
import subprocess
import re
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "datasets/independent_validation_data.csv"
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

df = pd.read_csv(DATA_FILE)
groups = list(df.groupby(["noise_std", "trial"], sort=True))
base_trial = groups[0][1].sort_values("time_s").reset_index(drop=True)
clean_ecg = base_trial["ecg_mV"].to_numpy().copy()
clean_hr = np.repeat([69.76, 84.50, 75.00, 95.23, 80.00, 69.76], 1000)
clean_spo2 = np.full(6000, 98.0)

def run_qemu_detailed(ecg, hr, spo2, dt_ms, labels):
    with open(HEADER, "w") as f:
        f.write("#ifndef QEMU_DATASET_INDEPENDENT_H\n#define QEMU_DATASET_INDEPENDENT_H\n")
        f.write("#include <stdint.h>\n#define DATASET_SIZE 6000\n#define SAMPLE_RATE 100.0f\n\n")
        for name, ctype, vals, fmt in [
            ("ecg_data", "float", ecg, lambda x: f"{x:.8f}f"),
            ("sensor_hr", "float", hr, lambda x: f"{x:.2f}f"),
            ("spo2_data", "float", spo2, lambda x: f"{x:.2f}f"),
            ("dt_ms", "uint32_t", dt_ms, lambda x: str(int(x))),
            ("fault_ground_truth", "uint8_t", labels, lambda x: str(int(x))),
        ]:
            f.write(f"static const {ctype} {name}[DATASET_SIZE] = {{\n")
            for i in range(0, len(vals), 10):
                f.write("    " + ", ".join(fmt(x) for x in vals[i:i+10]) + ",\n")
            f.write("};\n\n")
        f.write("#endif\n")

    subprocess.run([
        "arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
        "-ffreestanding", "-nostdlib", "-O2",
        "-I", "firmware/include",
        "-Wl,-T,firmware/linker/mps2_an386.ld",
        "firmware/startup/startup.s",
        "firmware/src/main.c", "firmware/src/ecg_filter.c",
        "firmware/src/r_peak.c", "firmware/src/hr_estimator.c",
        "firmware/src/fault_detector.c", "-lgcc", "-o", str(ELF)
    ], check=True, capture_output=True)

    run = subprocess.run([
        "qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4",
        "-kernel", str(ELF), "-nographic",
        "-semihosting-config", "enable=on,target=native"
    ], capture_output=True, text=True, timeout=15)
    out = run.stdout + run.stderr

    def grab(pat):
        return int(re.search(pat, out).group(1))

    win_hrs = [int(m) / 100.0 for m in re.findall(r"HR x100 = (\d+)", out)]
    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return rpeaks, tp, fp, fn, tn, prec * 100, rec * 100, f1 * 100, fpr * 100, win_hrs

# 1. Expanded Front-End Faults: Baseline Wander + 50 Hz Hum & Gradual Ramp Drift (20 trials each)
t_arr = np.arange(6000) / 100.0
extra_rows = []
for f_type in ["baseline_wander_hum", "gradual_ramp_drift"]:
    metrics = []
    for seed in range(20):
        rng = np.random.default_rng(4000 + seed)
        ecg, hr, spo2 = clean_ecg.copy(), clean_hr.copy(), clean_spo2.copy()
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        labels = np.zeros(6000, dtype=np.uint8)
        if f_type == "baseline_wander_hum":
            # 0.35 Hz respiration wander (0.4 mV) + 50 Hz powerline interference (0.12 mV)
            ecg += 0.40 * np.sin(2 * np.pi * 0.35 * t_arr + seed) + 0.12 * np.sin(2 * np.pi * 50.0 * t_arr)
            idx = rng.choice(np.arange(100, 5900), size=150, replace=False)
            hr[idx] = clean_hr[idx] + 35.0
            labels[idx] = 1
        elif f_type == "gradual_ramp_drift":
            # Slow calibration drift starting at t=20s (+1.5 BPM/s) until exceeding 10 BPM threshold
            drift = np.zeros(6000)
            drift[2000:2250] = np.linspace(0, 37.5, 250)
            hr += drift
            labels[np.where(drift > 10.0)[0]] = 1
        rpeaks, tp, fp, fn, tn, prec, rec, f1, fpr, _ = run_qemu_detailed(ecg, hr, spo2, dt_ms, labels)
        metrics.append([rpeaks, tp, fp, fn, prec, rec, f1, fpr])
    m = np.mean(metrics, axis=0)
    extra_rows.append({
        "Fault_Type": f_type, "R_Peaks": round(m[0], 2),
        "TP": round(m[1], 1), "FP": round(m[2], 1), "FN": round(m[3], 1),
        "Precision": round(m[4], 2), "Recall": round(m[5], 2),
        "F1": round(m[6], 2), "FPR": round(m[7], 2)
    })

extra_df = pd.DataFrame(extra_rows)
extra_df.to_csv(ROOT / "results/qemu_extra_faults.csv", index=False)
print("--- TABLE VI: EXPANDED FRONT-END & DRIFT FAULTS IN QEMU ---")
print(extra_df.to_string(index=False))

# 2. Windowed HR Estimation MAE & RMSE across Noise Levels (Mitigated vs Unmitigated)
true_win_hr = np.array([69.76, 84.50, 75.00, 95.23, 80.00, 69.76])
hr_err_rows = []
for noise in [0.08, 0.15, 0.25, 0.35]:
    sub = df[df["noise_std"] == noise]
    maes, rmses = [], []
    for trial_no, grp in sub.groupby("trial"):
        grp = grp.sort_values("time_s").reset_index(drop=True)
        _, _, _, _, _, _, _, _, _, win_hrs = run_qemu_detailed(
            grp["ecg_mV"].to_numpy(), grp["heart_rate_bpm"].to_numpy(),
            np.full(6000, 98.0), np.full(6000, 10, dtype=np.uint32),
            grp["fault"].to_numpy(dtype=np.uint8)
        )
        if len(win_hrs) == 6:
            err = np.abs(np.array(win_hrs) - true_win_hr)
            maes.append(np.mean(err))
            rmses.append(np.sqrt(np.mean(err ** 2)))
    hr_err_rows.append({
        "Noise_mV": noise,
        "ECG_HR_MAE_BPM": round(np.mean(maes), 3),
        "ECG_HR_RMSE_BPM": round(np.mean(rmses), 3),
        "Max_Window_Err_BPM": round(np.max(maes), 3)
    })

hr_err_df = pd.DataFrame(hr_err_rows)
hr_err_df.to_csv(ROOT / "results/qemu_hr_estimation_accuracy.csv", index=False)
print("\n--- TABLE VII: WINDOWED ECG HR ESTIMATION ACCURACY ACROSS NOISE LEVELS ---")
print(hr_err_df.to_string(index=False))
