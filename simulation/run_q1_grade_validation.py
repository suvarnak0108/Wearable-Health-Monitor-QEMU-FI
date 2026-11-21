import numpy as np
import pandas as pd
import subprocess
import re
from pathlib import Path
from scipy.signal import resample, butter, lfilter
from scipy.stats import wilcoxon
import wfdb

ROOT = Path.cwd()
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"
NATIVE_BIN = ROOT / "firmware/native_trial.bin"

C_SRCS = [
    "firmware/src/main.c", "firmware/src/ecg_filter.c",
    "firmware/src/r_peak.c", "firmware/src/hr_estimator.c",
    "firmware/src/fault_detector.c"
]

def compile_targets():
    # ARM Cortex-M4 with -fstack-usage
    subprocess.run([
        "arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
        "-ffreestanding", "-nostdlib", "-O2", "-fstack-usage",
        "-I", "firmware/include",
        "-Wl,-T,firmware/linker/mps2_an386.ld",
        "firmware/startup/startup.s", *C_SRCS, "-lgcc", "-o", str(ELF)
    ], check=True, capture_output=True)

def write_header(ecg, hr, spo2, dt_ms, labels):
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

def parse_log(out):
    def grab(pat):
        m = re.search(pat, out)
        return int(m.group(1)) if m else 0
    hr_raw = np.array([int(m)/100.0 for m in re.findall(r"HR_RAW x100 = (\d+)", out)])
    hr_rr  = np.array([int(m)/100.0 for m in re.findall(r"HR_RR  x100 = (\d+)", out)])
    hr_mit = np.array([int(m)/100.0 for m in re.findall(r"HR_MIT x100 = (\d+)", out)])
    sqi    = np.array([int(m)/100.0 for m in re.findall(r"SQI    x100 = (\d+)", out)])
    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    lat = grab(r"LatencySamples:\s*(\d+)")
    return rpeaks, tp, fp, fn, tn, lat, hr_raw, hr_rr, hr_mit, sqi

def run_qemu(ecg, hr, spo2, dt_ms, labels, icount=False):
    write_header(ecg, hr, spo2, dt_ms, labels)
    compile_targets()
    cmd = ["qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4",
           "-kernel", str(ELF), "-nographic", "-semihosting-config", "enable=on,target=native"]
    if icount:
        cmd.extend(["-icount", "shift=0"])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return parse_log(res.stdout + res.stderr)

def bootstrap_ci(arr, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# Generate continuous non-step HR synthetic ECG (65 to 135 BPM)
def make_continuous_ecg(seed=0):
    rng = np.random.default_rng(seed)
    fs = 100.0
    t = np.arange(6000) / fs
    # Continuous smooth trajectory reaching 135 BPM (tests >125 BPM refractory!)
    true_hr = 95.0 + 35.0 * np.sin(2 * np.pi * t / 60.0) + 5.0 * np.cos(2 * np.pi * t / 17.0)
    phase = np.cumsum(true_hr / 60.0 / fs)
    beat_times = []
    for k in range(1, int(np.floor(phase[-1])) + 1):
        idx = int(np.searchsorted(phase, k))
        if idx < 6000:
            beat_times.append(idx / fs)
    ecg = np.zeros(6000)
    for bt in beat_times:
        dt = t - bt
        ecg += (0.12*np.exp(-((dt+0.16)/0.025)**2) - 0.15*np.exp(-((dt+0.035)/0.010)**2)
                + 1.00*np.exp(-(dt/0.012)**2) - 0.22*np.exp(-((dt-0.035)/0.010)**2)
                + 0.28*np.exp(-((dt-0.22)/0.045)**2))
    # Independent auxiliary HR channel with realistic PPG lag (1.5 s) + jitter
    aux_hr = np.roll(true_hr, 150) + rng.normal(0, 1.8, size=6000)
    aux_hr[:150] = true_hr[:150]
    spo2 = np.clip(98.0 + rng.normal(0, 0.25, size=6000), 95.0, 100.0)
    win_true_hr = np.array([np.mean(true_hr[w*1000:(w+1)*1000]) for w in range(6)])
    return ecg, aux_hr, spo2, win_true_hr, len(beat_times)

# 1. PHYSIONET NSTDB REALISTIC NOISE ABLATION (em, ma, bw)
print("Downloading/loading PhysioNet NSTDB noise records (em, ma, bw)...")
nstdb_em = resample(wfdb.rdrecord("em", pn_dir="nstdb", sampto=21600).p_signal[:, 0], 6000)
nstdb_ma = resample(wfdb.rdrecord("ma", pn_dir="nstdb", sampto=21600).p_signal[:, 0], 6000)
nstdb_bw = resample(wfdb.rdrecord("bw", pn_dir="nstdb", sampto=21600).p_signal[:, 0], 6000)
nstdb_em = nstdb_em / np.std(nstdb_em)
nstdb_ma = nstdb_ma / np.std(nstdb_ma)
nstdb_bw = nstdb_bw / np.std(nstdb_bw)

nstdb_rows = []
for label, noise_vec, amp in [
    ("Clean Continuous HR (65-135 BPM)", np.zeros(6000), 0.02),
    ("NSTDB Baseline Wander (0.25 mV)", nstdb_bw, 0.25),
    ("NSTDB Muscle Artifact (0.20 mV)", nstdb_ma, 0.20),
    ("NSTDB Electrode Motion (0.25 mV)", nstdb_em, 0.25),
    ("NSTDB Electrode Motion (0.35 mV)", nstdb_em, 0.35),
]:
    maes_raw, maes_rr, maes_mit, f1s, fprs = [], [], [], [], []
    for trial in range(20):
        ecg0, aux_hr0, spo2_0, w_true, n_beats = make_continuous_ecg(seed=500 + trial)
        shift = (trial * 293) % 6000
        ecg = ecg0 + amp * np.roll(noise_vec, shift)
        hr = aux_hr0.copy()
        spo2 = spo2_0.copy()
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        gt = np.zeros(6000, dtype=np.uint8)

        # Inject independent physical faults (150 samples)
        rng = np.random.default_rng(700 + trial)
        f_idx = rng.choice(np.arange(100, 5900), size=150, replace=False)
        hr[f_idx] += rng.uniform(18.0, 40.0, size=150)
        gt[f_idx] = 1

        rp, tp, fp, fn, tn, lat, h_raw, h_rr, h_mit, sqi = run_qemu(ecg, hr, spo2, dt_ms, gt)
        prec = tp / (tp + fp) * 100 if (tp + fp) else 100.0
        rec = tp / (tp + fn) * 100 if (tp + fn) else 100.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 100.0
        fpr = fp / (fp + tn) * 100 if (fp + tn) else 0.0
        f1s.append(f1); fprs.append(fpr)
        maes_raw.append(np.mean(np.abs(h_raw - w_true)))
        maes_rr.append(np.mean(np.abs(h_rr - w_true)))
        maes_mit.append(np.mean(np.abs(h_mit - w_true)))

    lo, hi = bootstrap_ci(f1s)
    nstdb_rows.append({
        "Condition": label,
        "F1_Mean_SD": f"{np.mean(f1s):.2f} ± {np.std(f1s, ddof=1):.2f}",
        "F1_Boot95CI": f"[{lo:.2f}, {hi:.2f}]",
        "FPR_Mean_SD": f"{np.mean(fprs):.2f} ± {np.std(fprs, ddof=1):.2f}",
        "Raw_MAE": f"{np.mean(maes_raw):.2f}",
        "RR_Filt_MAE": f"{np.mean(maes_rr):.2f}",
        "Mitigated_MAE": f"{np.mean(maes_mit):.2f}"
    })

print("\n--- TABLE A: NON-CIRCULAR CONTINUOUS-HR + PHYSIONET NSTDB RESULTS (N=20) ---")
print(pd.DataFrame(nstdb_rows).to_string(index=False))

# 2. BOUNDARY DRIFT PROBING (8, 9, 10, 11, 12 BPM around 10 BPM threshold)
bound_rows = []
for drift_bpm in [8.0, 9.0, 10.0, 11.0, 12.0]:
    ecg0, aux_hr0, spo2_0, _, _ = make_continuous_ecg(seed=42)
    hr = aux_hr0.copy()
    gt = np.zeros(6000, dtype=np.uint8)
    hr[2000:3500] += drift_bpm
    if drift_bpm > 10.0:
        gt[2000:3500] = 1
    rp, tp, fp, fn, tn, lat, _, _, _, _ = run_qemu(ecg0, hr, spo2_0, np.full(6000, 10, dtype=np.uint32), gt)
    det_rate = (tp + fp) / 1500.0 * 100.0
    bound_rows.append({"Injected_Drift_BPM": drift_bpm, "Flagged_Pct": f"{det_rate:.2f}%", "TP": tp, "FP": fp})

print("\n--- TABLE B: BOUNDARY DRIFT SENSITIVITY AROUND 10 BPM THRESHOLD ---")
print(pd.DataFrame(bound_rows).to_string(index=False))

# 3. MIT-BIH 10-PATIENT ABLATION (Raw vs RR-Filtered vs Hold-Last-HR vs Simple Median-HR)
records = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
beat_syms = {"N", "L", "R", "A", "V", "F", "e", "j", "E", "J"}
mit_rows = []
for rec_id in records:
    rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=21600)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb", sampto=21600)
    ecg = resample(rec.p_signal[:, 0], 6000)
    ecg = (ecg - np.median(ecg)) / np.percentile(np.abs(ecg - np.median(ecg)), 99)
    true_r = np.array([int(round(s * 100.0 / 360.0)) for s, sym in zip(ann.sample, ann.symbol) if sym in beat_syms])
    true_r = true_r[(true_r > 0) & (true_r < 6000)]

    # True window HRs from annotations
    w_true = []
    for w in range(6):
        wp = true_r[(true_r >= w*1000) & (true_r < (w+1)*1000)]
        w_true.append(60.0 * 100.0 / np.median(np.diff(wp)) if len(wp) >= 2 else 75.0)
    w_true = np.array(w_true)

    # Inject triple compound fault
    rng = np.random.default_rng(300 + int(rec_id))
    ecg_f = ecg + rng.normal(0, 0.15, size=6000)
    hr_f = np.repeat(w_true, 1000) + rng.normal(0, 1.5, size=6000)
    spo2_f = np.full(6000, 98.0)
    dt_f = np.full(6000, 10, dtype=np.uint32)
    gt = np.zeros(6000, dtype=np.uint8)
    idx_m = rng.choice(np.arange(100, 5900), size=100, replace=False)
    dt_f[idx_m] = 25; gt[idx_m] = 1
    idx_d = rng.choice(np.setdiff1d(np.arange(100, 5900), idx_m), size=100, replace=False)
    hr_f[idx_d] += 25.0; gt[idx_d] = 1

    rp, tp, fp, fn, tn, lat, h_raw, h_rr, h_mit, sqi = run_qemu(ecg_f, hr_f, spo2_f, dt_f, gt)
    # Simple median-HR baseline (cumulative median of raw HR)
    h_med_base = np.array([np.median(h_raw[:w+1]) for w in range(6)])
    mit_rows.append({
        "Record": rec_id,
        "True_Beats": len(true_r),
        "Det_Peaks": rp,
        "Raw_MAE": np.mean(np.abs(h_raw - w_true)),
        "Median_Base_MAE": np.mean(np.abs(h_med_base - w_true)),
        "RR_Filt_MAE": np.mean(np.abs(h_rr - w_true)),
        "Mitigated_MAE": np.mean(np.abs(h_mit - w_true)),
        "SQI": np.mean(sqi)
    })

mit_df = pd.DataFrame(mit_rows)
print("\n--- TABLE C: MIT-BIH 10-PATIENT ABLATION (FULL-WAVE RECTIFICATION + 300ms REFRACTORY) ---")
print(mit_df.to_string(index=False))
diffs = mit_df["Raw_MAE"] - mit_df["Mitigated_MAE"]
nz = diffs[np.abs(diffs) > 1e-6]
w_stat, p_val = wilcoxon(nz)
print(f"Mean Raw={mit_df['Raw_MAE'].mean():.3f} -> Mitigated={mit_df['Mitigated_MAE'].mean():.3f} | "
      f"Median Raw={mit_df['Raw_MAE'].median():.3f} -> Mitigated={mit_df['Mitigated_MAE'].median():.3f} | "
      f"Effective n={len(nz)}, W={w_stat}, p={p_val:.4f}")

# 4. STACK USAGE (-fstack-usage) & ELF SIZE
print("\n--- TABLE D: GCC -fstack-usage & ELF FOOTPRINT ---")
for su in sorted(ROOT.glob("*.su")):
    print(su.read_text().strip())
print(subprocess.run(["arm-none-eabi-size", "-A", str(ELF)], capture_output=True, text=True).stdout)
