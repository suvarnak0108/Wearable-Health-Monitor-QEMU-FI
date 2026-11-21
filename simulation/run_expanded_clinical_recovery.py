import numpy as np
import pandas as pd
import subprocess
import re
from pathlib import Path
from scipy.signal import resample
import wfdb

ROOT = Path.cwd()
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

def run_qemu_clinical_trial(ecg, hr, spo2, dt_ms, labels):
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

    raw_hrs = [int(m) / 100.0 for m in re.findall(r"HR x100 = (\d+)", out)]
    mit_hrs = [int(m) / 100.0 for m in re.findall(r"Mitigated x100 = (\d+)", out)]
    sqis = [int(m) / 100.0 for m in re.findall(r"SQI x100 = (\d+)", out)]

    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) * 100 if (tp + fp) else 100.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 100.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 100.0
    fpr = fp / (fp + tn) * 100 if (fp + tn) else 0.0
    return rpeaks, tp, fp, fn, prec, rec, f1, fpr, raw_hrs, mit_hrs, sqis

# 10 Diverse Clinical Records from PhysioNet MIT-BIH Arrhythmia Database
records = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
scenarios = ["clean_clinical", "noise_020_clinical", "burst_stale_clinical", "triple_compound_clinical"]

all_rows = []
for rec_id in records:
    rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=21600)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb", sampto=21600)

    ecg_base = resample(rec.p_signal[:, 0], 6000)
    ecg_base = (ecg_base - np.median(ecg_base)) / np.percentile(np.abs(ecg_base - np.median(ecg_base)), 99)

    beat_syms = {"N", "L", "R", "A", "V", "F", "e", "j", "E", "J"}
    true_r = np.array([int(round(s * 100.0 / 360.0)) for s, sym in zip(ann.sample, ann.symbol) if sym in beat_syms])
    true_r = true_r[(true_r > 0) & (true_r < 6000)]

    true_win_hrs = []
    hr_base = np.zeros(6000)
    for w in range(6):
        wp = true_r[(true_r >= w*1000) & (true_r < (w+1)*1000)]
        whr = 6000.0 / np.median(np.diff(wp)) if len(wp) > 1 else 75.0
        true_win_hrs.append(whr)
        hr_base[w*1000:(w+1)*1000] = whr
    true_win_hrs = np.array(true_win_hrs)

    for scen in scenarios:
        rng = np.random.default_rng(700 + int(rec_id))
        ecg = ecg_base.copy()
        hr = hr_base.copy()
        spo2 = np.full(6000, 98.0)
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        labels = np.zeros(6000, dtype=np.uint8)

        if scen == "noise_020_clinical":
            ecg += rng.normal(0, 0.20, size=6000)
            idx = rng.choice(np.arange(100, 5900), size=150, replace=False)
            hr[idx] += 35.0
            labels[idx] = 1
        elif scen == "burst_stale_clinical":
            dt_ms[1200:1275] = 60
            labels[1200:1275] = 1
            ecg[3500:3579] = ecg[3500]
            hr[3500:3579] = hr[3500]
            spo2[3500:3579] = 98.0
            labels[3504:3579] = 1
        elif scen == "triple_compound_clinical":
            ecg += rng.normal(0, 0.15, size=6000)
            idx_m = rng.choice(np.arange(10, 5990), size=75, replace=False)
            dt_ms[idx_m] = 20
            labels[idx_m] = 1
            rem = np.where(labels == 0)[0][10:-10]
            idx_d = rng.choice(rem, size=75, replace=False)
            hr[idx_d] += 35.0
            labels[idx_d] = 1

        rpeaks, tp, fp, fn, prec, rec_pct, f1, fpr, raw_hrs, mit_hrs, sqis = run_qemu_clinical_trial(
            ecg, hr, spo2, dt_ms, labels
        )
        raw_mae = np.mean(np.abs(np.array(raw_hrs) - true_win_hrs)) if len(raw_hrs) == 6 else np.nan
        mit_mae = np.mean(np.abs(np.array(mit_hrs) - true_win_hrs)) if len(mit_hrs) == 6 else np.nan

        all_rows.append({
            "Record": rec_id, "Scenario": scen,
            "True_Beats": len(true_r), "Det_Peaks": rpeaks,
            "Prec": round(prec, 2), "Rec": round(rec_pct, 2),
            "F1": round(f1, 2), "FPR": round(fpr, 2),
            "Raw_HR_MAE": round(raw_mae, 2), "Mitigated_HR_MAE": round(mit_mae, 2),
            "Mean_SQI": round(np.mean(sqis), 2)
        })

full_clin_df = pd.DataFrame(all_rows)
full_clin_df.to_csv(ROOT / "results/qemu_40_clinical_trials.csv", index=False)

summary_by_scen = full_clin_df.groupby("Scenario")[
    ["Prec", "Rec", "F1", "FPR", "Raw_HR_MAE", "Mitigated_HR_MAE", "Mean_SQI"]
].mean().reset_index()
summary_by_scen.to_csv(ROOT / "results/qemu_clinical_scenario_summary.csv", index=False)

print("\n--- TABLE X: 40-TRIAL CLINICAL MIT-BIH SUMMARY ACROSS 4 SCENARIOS (10 PATIENTS) ---")
print(summary_by_scen.to_string(index=False))

print("\n--- PER-PATIENT CLINICAL BREAKDOWN (TRIPLE COMPOUND FAULT) ---")
print(full_clin_df[full_clin_df["Scenario"] == "triple_compound_clinical"].to_string(index=False))

nm_out = subprocess.run(["arm-none-eabi-nm", "--size-sort", "--print-size", str(ELF)], capture_output=True, text=True)
print("\n--- UPDATED CORTEX-M4 CORE CODE SIZE ---")
for line in nm_out.stdout.splitlines():
    if any(k in line for k in ["ECG_", "RPeak_", "HR_", "Fault_", "main"]):
        print(line)
