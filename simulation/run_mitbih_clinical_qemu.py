import numpy as np
import pandas as pd
import subprocess
import re
from pathlib import Path
from scipy.signal import resample

ROOT = Path.cwd()
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

def run_qemu_on_clinical(ecg, hr, spo2, dt_ms, labels):
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

    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) * 100 if (fp + tn) else 0.0
    return rpeaks, tp, fp, fn, prec, rec, f1, fpr

import wfdb

# 5 diverse MIT-BIH Arrhythmia Database records (Normal, Noisy, Baseline Wander, Arrhythmia)
records = ["100", "101", "103", "105", "119"]
clinical_rows = []

for rec_id in records:
    rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=21600)  # 60s at 360 Hz
    ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb", sampto=21600)
    
    raw_360 = rec.p_signal[:, 0]
    ecg_100 = resample(raw_360, 6000)
    # Normalize clinical amplitude to ~1.0 mV R-peak scale matching wearable AFE gain
    ecg_100 = (ecg_100 - np.median(ecg_100)) / np.percentile(np.abs(ecg_100 - np.median(ecg_100)), 99)

    # True beat annotations (exclude non-beat symbols)
    beat_syms = {"N", "L", "R", "A", "V", "F", "e", "j", "E", "J"}
    true_r_100 = np.array([int(round(s * 100.0 / 360.0)) for s, sym in zip(ann.sample, ann.symbol) if sym in beat_syms])
    true_r_100 = true_r_100[(true_r_100 > 0) & (true_r_100 < 6000)]

    # Build true window HR from clinical R-R intervals
    hr_100 = np.zeros(6000)
    for w in range(6):
        w_peaks = true_r_100[(true_r_100 >= w*1000) & (true_r_100 < (w+1)*1000)]
        w_hr = 6000.0 / np.median(np.diff(w_peaks)) if len(w_peaks) > 1 else 75.0
        hr_100[w*1000:(w+1)*1000] = w_hr

    # Test compound fault injection (missing samples + out-of-range + in-range HR drift) on real clinical ECG
    rng = np.random.default_rng(500 + int(rec_id))
    spo2_100 = np.full(6000, 98.0)
    dt_ms = np.full(6000, 10, dtype=np.uint32)
    labels = np.zeros(6000, dtype=np.uint8)

    idx_m = rng.choice(np.arange(10, 5990), size=50, replace=False)
    dt_ms[idx_m] = 20
    labels[idx_m] = 1

    rem = np.where(labels == 0)[0][10:-10]
    idx_o = rng.choice(rem, size=50, replace=False)
    hr_100[idx_o[:25]] = 225.0
    spo2_100[idx_o[25:]] = 55.0
    labels[idx_o] = 1

    rem2 = np.where(labels == 0)[0][10:-10]
    idx_d = rng.choice(rem2, size=50, replace=False)
    hr_100[idx_d] += 35.0
    labels[idx_d] = 1

    rpeaks, tp, fp, fn, prec, rec_pct, f1, fpr = run_qemu_on_clinical(ecg_100, hr_100, spo2_100, dt_ms, labels)
    clinical_rows.append({
        "MITBIH_Record": rec_id,
        "True_Beats": len(true_r_100),
        "Detected_Peaks": rpeaks,
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(prec, 2),
        "Recall": round(rec_pct, 2),
        "F1": round(f1, 2),
        "FPR": round(fpr, 2)
    })

clin_df = pd.DataFrame(clinical_rows)
clin_df.to_csv(ROOT / "results/qemu_mitbih_clinical_validation.csv", index=False)
print("\n--- TABLE IX: REAL CLINICAL MIT-BIH ARRHYTHMIA VALIDATION IN QEMU ---")
print(clin_df.to_string(index=False))
