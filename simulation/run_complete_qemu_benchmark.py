import pandas as pd
import numpy as np
import subprocess
import re
from pathlib import Path

ROOT = Path.cwd()
DATA_FILE = ROOT / "datasets/independent_validation_data.csv"
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

def compile_and_run_qemu(ecg, hr, spo2, dt_ms, labels):
    with open(HEADER, "w") as f:
        f.write("#ifndef QEMU_DATASET_INDEPENDENT_H\n")
        f.write("#define QEMU_DATASET_INDEPENDENT_H\n")
        f.write("#include <stdint.h>\n")
        f.write("#define DATASET_SIZE 6000\n")
        f.write("#define SAMPLE_RATE 100.0f\n\n")

        def write_arr(name, ctype, vals, fmt):
            f.write(f"static const {ctype} {name}[DATASET_SIZE] = {{\n")
            for i in range(0, len(vals), 10):
                f.write("    " + ", ".join(fmt(x) for x in vals[i:i+10]) + ",\n")
            f.write("};\n\n")

        write_arr("ecg_data", "float", ecg, lambda x: f"{x:.8f}f")
        write_arr("sensor_hr", "float", hr, lambda x: f"{x:.2f}f")
        write_arr("spo2_data", "float", spo2, lambda x: f"{x:.2f}f")
        write_arr("dt_ms", "uint32_t", dt_ms, lambda x: str(int(x)))
        write_arr("fault_ground_truth", "uint8_t", labels, lambda x: str(int(x)))
        f.write("#endif\n")

    cmd_build = [
        "arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
        "-ffreestanding", "-nostdlib", "-Wall", "-Wextra", "-O2",
        "-I", "firmware/include",
        "-Wl,-T,firmware/linker/mps2_an386.ld",
        "firmware/startup/startup.s",
        "firmware/src/main.c",
        "firmware/src/ecg_filter.c",
        "firmware/src/r_peak.c",
        "firmware/src/hr_estimator.c",
        "firmware/src/fault_detector.c",
        "-lgcc", "-o", str(ELF)
    ]
    b = subprocess.run(cmd_build, capture_output=True, text=True)
    if b.returncode != 0:
        raise RuntimeError(f"Build failed:\n{b.stderr}")

    cmd_qemu = [
        "qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4",
        "-kernel", str(ELF), "-nographic",
        "-semihosting-config", "enable=on,target=native"
    ]
    r = subprocess.run(cmd_qemu, capture_output=True, text=True, timeout=15)
    out = r.stdout + r.stderr

    def grab(pat):
        m = re.search(pat, out)
        if not m:
            raise RuntimeError(f"Missing {pat} in:\n{out}")
        return int(m.group(1))

    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp = grab(r"TP:\s*(\d+)")
    fp = grab(r"FP:\s*(\d+)")
    fn = grab(r"FN:\s*(\d+)")
    tn = grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return rpeaks, tp, fp, fn, tn, prec, rec, f1, fpr

df = pd.read_csv(DATA_FILE)
groups = list(df.groupby(["noise_std", "trial"], sort=True))

# Part A: Re-run 80 Noise Trials with Adaptive Firmware
noise_results = []
for idx, ((noise, trial_no), trial) in enumerate(groups, 1):
    trial = trial.sort_values("time_s").reset_index(drop=True)
    ecg = trial["ecg_mV"].to_numpy()
    hr = trial["heart_rate_bpm"].to_numpy()
    spo2 = trial["spo2_pct"].to_numpy() if "spo2_pct" in trial.columns else np.full(6000, 98.0)
    dt_ms = np.full(6000, 10, dtype=np.uint32)
    labels = trial["fault"].to_numpy(dtype=np.uint8)

    rpeaks, tp, fp, fn, tn, prec, rec, f1, fpr = compile_and_run_qemu(ecg, hr, spo2, dt_ms, labels)
    noise_results.append({
        "noise_std": noise, "trial": trial_no, "r_peaks": rpeaks,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": prec, "recall": rec, "F1": f1, "FPR": fpr
    })
    if idx % 20 == 0:
        print(f"Completed {idx}/80 adaptive noise trials (Noise={noise:.2f} -> F1={f1*100:.2f}%)")

noise_df = pd.DataFrame(noise_results)
noise_df.to_csv(ROOT / "results/qemu_adaptive_80_trials.csv", index=False)
noise_sum = noise_df.groupby("noise_std")[["precision", "recall", "F1", "FPR", "r_peaks"]].mean().reset_index()
noise_sum.to_csv(ROOT / "results/qemu_adaptive_summary.csv", index=False)
print("\n--- TABLE I: ADAPTIVE FIRMWARE ACROSS NOISE LEVELS (QEMU) ---")
print(noise_sum.to_string(index=False))

# Part B: Run 5 Non-Noise Fault Classes (20 trials each, 6000 samples)
base_trial = groups[0][1].sort_values("time_s").reset_index(drop=True)
clean_ecg = base_trial["ecg_mV"].to_numpy().copy()
clean_hr = np.repeat([69.76, 84.50, 75.00, 95.23, 80.00, 69.76], 1000)
clean_spo2 = np.full(6000, 98.0)

fault_scenarios = ["missing_samples", "comm_burst_loss", "out_of_range", "stale_data", "in_range_hr_drift"]
fault_rows = []

for f_type in fault_scenarios:
    trial_metrics = []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        ecg = clean_ecg.copy()
        hr = clean_hr.copy()
        spo2 = clean_spo2.copy()
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        labels = np.zeros(6000, dtype=np.uint8)

        if f_type == "missing_samples":
            idx = rng.choice(np.arange(10, 5990), size=150, replace=False)
            dt_ms[idx] = 20
            labels[idx] = 1
        elif f_type == "comm_burst_loss":
            for start in [1000 + seed * 10, 2800 + seed * 10, 4500 + seed * 10]:
                dt_ms[start:start+50] = 60
                labels[start:start+50] = 1
        elif f_type == "out_of_range":
            idx = rng.choice(np.arange(10, 5990), size=150, replace=False)
            hr[idx[:75]] = rng.choice([20.0, 225.0], size=75)
            spo2[idx[75:]] = rng.choice([55.0, 105.0], size=75)
            labels[idx] = 1
        elif f_type == "stale_data":
            # 3 bursts of 54 samples; first 4 establish freeze, remaining 50 trigger fault (150 total)
            for start in [900 + seed * 10, 2600 + seed * 10, 4300 + seed * 10]:
                ecg[start:start+54] = ecg[start]
                hr[start:start+54] = 76.0
                spo2[start:start+54] = 97.0
                labels[start+4:start+54] = 1
        elif f_type == "in_range_hr_drift":
            idx = rng.choice(np.arange(100, 5900), size=150, replace=False)
            hr[idx] = clean_hr[idx] + 35.0
            labels[idx] = 1

        rpeaks, tp, fp, fn, tn, prec, rec, f1, fpr = compile_and_run_qemu(ecg, hr, spo2, dt_ms, labels)
        trial_metrics.append([tp, fp, fn, tn, prec * 100, rec * 100, f1 * 100, fpr * 100])

    m = np.mean(trial_metrics, axis=0)
    fault_rows.append({
        "Fault_Scenario": f_type,
        "TP": round(m[0], 1), "FP": round(m[1], 1), "FN": round(m[2], 1), "TN": round(m[3], 1),
        "Precision_pct": round(m[4], 2), "Recall_pct": round(m[5], 2),
        "F1_pct": round(m[6], 2), "FPR_pct": round(m[7], 2)
    })

fault_df = pd.DataFrame(fault_rows)
fault_df.to_csv(ROOT / "results/qemu_all_faults_summary.csv", index=False)
print("\n--- TABLE II: MULTI-FAULT CLASSIFICATION IN QEMU (20 TRIALS EACH, 6000 SAMPLES) ---")
print(fault_df.to_string(index=False))

size_out = subprocess.run(["arm-none-eabi-size", str(ELF)], capture_output=True, text=True)
print("\n--- CORTEX-M4 FIRMWARE MEMORY FOOTPRINT ---")
print(size_out.stdout)
