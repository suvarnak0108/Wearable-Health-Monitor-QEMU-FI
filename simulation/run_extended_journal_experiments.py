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

def run_qemu_trial(ecg, hr, spo2, dt_ms, labels):
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
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tp, fp, fn, tn, prec * 100, rec * 100, f1 * 100, fpr * 100, win_hrs

# 1. Fault-Density Sensitivity Sweep (1%, 2.5%, 5%, 10%, 20%)
density_rows = []
for pct in [1.0, 2.5, 5.0, 10.0, 20.0]:
    n_faults = int(6000 * pct / 100.0)
    for f_type in ["missing_samples", "out_of_range", "in_range_hr_drift"]:
        metrics = []
        for seed in range(10):
            rng = np.random.default_rng(2000 + seed)
            ecg, hr, spo2 = clean_ecg.copy(), clean_hr.copy(), clean_spo2.copy()
            dt_ms = np.full(6000, 10, dtype=np.uint32)
            labels = np.zeros(6000, dtype=np.uint8)
            idx = rng.choice(np.arange(10, 5990), size=n_faults, replace=False)
            if f_type == "missing_samples":
                dt_ms[idx] = 20
            elif f_type == "out_of_range":
                hr[idx[:n_faults//2]] = 225.0
                spo2[idx[n_faults//2:]] = 55.0
            elif f_type == "in_range_hr_drift":
                hr[idx] = clean_hr[idx] + 35.0
            labels[idx] = 1
            metrics.append(run_qemu_trial(ecg, hr, spo2, dt_ms, labels)[:8])
        m = np.mean(metrics, axis=0)
        density_rows.append({
            "Fault_Type": f_type, "Density_pct": pct,
            "Precision": round(m[4], 2), "Recall": round(m[5], 2),
            "F1": round(m[6], 2), "FPR": round(m[7], 2)
        })

density_df = pd.DataFrame(density_rows)
density_df.to_csv(ROOT / "results/qemu_density_sweep.csv", index=False)
print("--- TABLE III: FAULT-DENSITY SENSITIVITY SWEEP IN QEMU ---")
print(density_df.to_string(index=False))

# 2. Compound / Concurrent Multi-Fault Stress Scenarios (20 trials each)
compound_rows = []
for scenario in ["noise_plus_missing", "burst_plus_stale", "noise_plus_drift", "triple_compound"]:
    metrics = []
    for seed in range(20):
        rng = np.random.default_rng(3000 + seed)
        ecg, hr, spo2 = clean_ecg.copy(), clean_hr.copy(), clean_spo2.copy()
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        labels = np.zeros(6000, dtype=np.uint8)

        if scenario in ["noise_plus_missing", "noise_plus_drift", "triple_compound"]:
            ecg += rng.normal(0, 0.15, size=6000)
        if scenario in ["noise_plus_missing", "triple_compound"]:
            idx_m = rng.choice(np.arange(10, 5990), size=100, replace=False)
            dt_ms[idx_m] = 20
            labels[idx_m] = 1
        if scenario == "burst_plus_stale":
            dt_ms[1200:1275] = 60
            labels[1200:1275] = 1
            ecg[3500:3579] = ecg[3500]
            hr[3500:3579] = 75.0
            spo2[3500:3579] = 98.0
            labels[3504:3579] = 1
        if scenario in ["noise_plus_drift", "triple_compound"]:
            idx_d = rng.choice(np.where(labels == 0)[0][10:-10], size=100, replace=False)
            hr[idx_d] = clean_hr[idx_d] + 35.0
            labels[idx_d] = 1

        metrics.append(run_qemu_trial(ecg, hr, spo2, dt_ms, labels)[:8])
    m = np.mean(metrics, axis=0)
    compound_rows.append({
        "Compound_Scenario": scenario,
        "TP": round(m[0], 1), "FP": round(m[1], 1), "FN": round(m[2], 1),
        "Precision": round(m[4], 2), "Recall": round(m[5], 2),
        "F1": round(m[6], 2), "FPR": round(m[7], 2)
    })

compound_df = pd.DataFrame(compound_rows)
compound_df.to_csv(ROOT / "results/qemu_compound_faults.csv", index=False)
print("\n--- TABLE IV: CONCURRENT / COMPOUND MULTI-FAULT STRESS TESTS ---")
print(compound_df.to_string(index=False))

# 3. Per-Module Memory Footprint Breakdown
nm_out = subprocess.run(["arm-none-eabi-nm", "--size-sort", "--print-size", str(ELF)], capture_output=True, text=True)
print("\n--- TABLE V: PER-SYMBOL CORTEX-M4 FOOTPRINT ---")
for line in nm_out.stdout.splitlines():
    if any(k in line for k in ["ECG_", "RPeak_", "HR_", "Fault_", "main", "ecg_data", "sensor_hr"]):
        print(line)
