import numpy as np
import pandas as pd
import subprocess
import re
from pathlib import Path
from scipy.signal import resample, butter, filtfilt, lfilter
from scipy.stats import wilcoxon
import wfdb

ROOT = Path.cwd()
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

def compile_firmware():
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

def run_qemu_trial(ecg, hr, spo2, dt_ms, labels, count_icount=False):
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

    compile_firmware()
    cmd = [
        "qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4",
        "-kernel", str(ELF), "-nographic",
        "-semihosting-config", "enable=on,target=native"
    ]
    if count_icount:
        cmd.extend(["-icount", "shift=0"])

    run = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    out = run.stdout + run.stderr

    def grab(pat):
        m = re.search(pat, out)
        return int(m.group(1)) if m else 0

    raw_hrs = [int(m) / 100.0 for m in re.findall(r"HR x100 = (\d+)", out)]
    mit_hrs = [int(m) / 100.0 for m in re.findall(r"Mitigated x100 = (\d+)", out)]
    sqis = [int(m) / 100.0 for m in re.findall(r"SQI x100 = (\d+)", out)]

    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    prec = tp / (tp + fp) * 100 if (tp + fp) else 100.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 100.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 100.0
    fpr = fp / (fp + tn) * 100 if (fp + tn) else 0.0
    return rpeaks, tp, fp, fn, tn, prec, rec, f1, fpr, raw_hrs, mit_hrs, sqis

# Load or synthesize base clean dataset (6000 samples @ 100 Hz, 6 regimes)
true_win_hrs = np.array([70.0, 84.0, 75.0, 95.0, 80.0, 70.0])
csv_path = ROOT / "data/processed/independent_validation_data.csv"
if csv_path.exists():
    base_df = pd.read_csv(csv_path)
    base_ecg = base_df["ecg"].to_numpy()
    base_hr = base_df["heart_rate"].to_numpy()
else:
    fs = 100.0
    base_ecg = np.zeros(6000)
    base_hr = np.zeros(6000)
    t_cursor = 0.0
    for w, hr_bpm in enumerate(true_win_hrs):
        base_hr[w*1000:(w+1)*1000] = hr_bpm
        rr = 60.0 / hr_bpm
        while t_cursor < (w + 1) * 10.0:
            t_arr = np.arange(w*1000, (w+1)*1000) / fs - t_cursor
            base_ecg[w*1000:(w+1)*1000] += (
                0.12 * np.exp(-((t_arr + 0.16) / 0.025)**2)
                - 0.15 * np.exp(-((t_arr + 0.035) / 0.010)**2)
                + 1.00 * np.exp(-((t_arr) / 0.012)**2)
                - 0.22 * np.exp(-((t_arr - 0.035) / 0.010)**2)
                + 0.28 * np.exp(-((t_arr - 0.22) / 0.045)**2)
            )
            t_cursor += rr

# 1. RANDOMIZED MONTE CARLO TRIALS (Mean ± SD and 95% CI)
# Fix #3 & #8: Randomized fault counts/positions per seed + fix Window Error definition
noise_stats = []
for sigma in [0.08, 0.15, 0.25, 0.35]:
    f1s, fprs, maes, rmses, max_errs, peak_cnts = [], [], [], [], [], []
    for seed in range(20):
        rng = np.random.default_rng(1000 + int(sigma * 1000) + seed)
        ecg = base_ecg + rng.normal(0, sigma, size=6000)
        hr = base_hr.copy()
        spo2 = np.full(6000, 98.0)
        dt_ms = np.full(6000, 10, dtype=np.uint32)
        labels = np.zeros(6000, dtype=np.uint8)

        # Randomize fault count (120 to 180 samples) & positions per trial
        n_faults = int(rng.integers(120, 181))
        idx = rng.choice(np.arange(100, 5900), size=n_faults, replace=False)
        hr[idx] += 35.0
        labels[idx] = 1

        rp, tp, fp, fn, tn, prec, rec, f1, fpr, raw_hrs, mit_hrs, _ = run_qemu_trial(ecg, hr, spo2, dt_ms, labels)
        f1s.append(f1)
        fprs.append(fpr)
        peak_cnts.append(rp)
        if len(raw_hrs) == 6:
            errs = np.abs(np.array(raw_hrs) - true_win_hrs)
            maes.append(np.mean(errs))
            rmses.append(np.sqrt(np.mean(errs**2)))
            max_errs.append(np.max(errs))

    def fmt_ci(arr):
        m, s = np.mean(arr), np.std(arr, ddof=1)
        ci = 1.96 * s / np.sqrt(len(arr))
        return f"{m:.2f} ± {s:.2f} [{m-ci:.2f}, {m+ci:.2f}]"

    noise_stats.append({
        "Noise_mV": sigma,
        "R_Peaks_Mean_SD": f"{np.mean(peak_cnts):.2f} ± {np.std(peak_cnts, ddof=1):.2f}",
        "F1_Mean_SD_CI": fmt_ci(f1s),
        "FPR_Mean_SD_CI": fmt_ci(fprs),
        "MAE_Mean_SD": f"{np.mean(maes):.3f} ± {np.std(maes, ddof=1):.3f}",
        "RMSE_Mean_SD": f"{np.mean(rmses):.3f} ± {np.std(rmses, ddof=1):.3f}",
        "Worst_Trial_Window_Err": f"{np.max(max_errs):.3f}"
    })

noise_ci_df = pd.DataFrame(noise_stats)
noise_ci_df.to_csv(ROOT / "results/reviewer_noise_stats_ci.csv", index=False)
print("\n--- REVISED TABLE: RANDOMIZED NOISE MONTE CARLO (MEAN ± SD & 95% CI, N=20) ---")
print(noise_ci_df.to_string(index=False))

# 2. CLASSICAL BASELINES (Pan-Tompkins & Elgendi) + AAMI EC57 BEAT METRICS (150 ms tolerance)
def detect_pan_tompkins(ecg, fs=100.0):
    b, a = butter(2, [5.0 / (fs/2), 15.0 / (fs/2)], btype="band")
    y = lfilter(b, a, ecg)
    dy = np.diff(y, prepend=y[0])
    sq = dy ** 2
    win = int(0.15 * fs)
    mwi = np.convolve(sq, np.ones(win)/win, mode="same")
    peaks = []
    spki, npki = 0.05, 0.01
    last_p = -100
    for i in range(1, len(mwi)-1):
        if mwi[i] > mwi[i-1] and mwi[i] >= mwi[i+1]:
            thresh = npki + 0.25 * (spki - npki)
            if (i - last_p) >= 36 and mwi[i] > thresh:
                peaks.append(i)
                spki = 0.125 * mwi[i] + 0.875 * spki
                last_p = i
            else:
                npki = 0.125 * mwi[i] + 0.875 * npki
    return np.array(peaks)

def detect_elgendi(ecg, fs=100.0):
    b, a = butter(2, [8.0 / (fs/2), 20.0 / (fs/2)], btype="band")
    y = lfilter(b, a, ecg)
    sq = y ** 2
    ma_qrs = np.convolve(sq, np.ones(int(0.12*fs))/int(0.12*fs), mode="same")
    ma_beat = np.convolve(sq, np.ones(int(0.60*fs))/int(0.60*fs), mode="same")
    thr = ma_beat + 0.08 * np.mean(sq)
    active = ma_qrs > thr
    peaks = []
    in_seg, start = False, 0
    for i, val in enumerate(active):
        if val and not in_seg:
            in_seg, start = True, i
        elif not val and in_seg:
            in_seg = False
            if (i - start) >= int(0.08 * fs):
                p = start + np.argmax(y[start:i])
                if not peaks or (p - peaks[-1]) >= 36:
                    peaks.append(p)
    return np.array(peaks)

def aami_ec57_metrics(true_peaks, det_peaks, tol_samples=15):
    matched_det = set()
    tp = 0
    for tp_idx in true_peaks:
        diffs = np.abs(det_peaks - tp_idx) if len(det_peaks) else np.array([])
        if len(diffs) and np.min(diffs) <= tol_samples:
            best = int(np.argmin(diffs))
            if best not in matched_det:
                matched_det.add(best)
                tp += 1
    fn = len(true_peaks) - tp
    fp = len(det_peaks) - tp
    se = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
    ppv = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
    return se, ppv, f1

# Evaluate on 10 MIT-BIH records (Clean & 0.20 mV Noise)
records = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
beat_syms = {"N", "L", "R", "A", "V", "F", "e", "j", "E", "J"}
baseline_rows = []

for cond in ["Clean MIT-BIH (10 Records)", "Noisy MIT-BIH 0.20 mV (10 Records)"]:
    pt_se, pt_ppv, pt_f1 = [], [], []
    el_se, el_ppv, el_f1 = [], [], []
    qr_se, qr_ppv, qr_f1 = [], [], []

    for rec_id in records:
        rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=21600)
        ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb", sampto=21600)
        ecg = resample(rec.p_signal[:, 0], 6000)
        ecg = (ecg - np.median(ecg)) / np.percentile(np.abs(ecg - np.median(ecg)), 99)
        if "0.20 mV" in cond:
            rng = np.random.default_rng(900 + int(rec_id))
            ecg = ecg + rng.normal(0, 0.20, size=6000)

        true_r = np.array([int(round(s * 100.0 / 360.0)) for s, sym in zip(ann.sample, ann.symbol) if sym in beat_syms])
        true_r = true_r[(true_r > 0) & (true_r < 6000)]

        # 1. Pan-Tompkins
        s1, p1, f1_1 = aami_ec57_metrics(true_r, detect_pan_tompkins(ecg))
        pt_se.append(s1); pt_ppv.append(p1); pt_f1.append(f1_1)

        # 2. Elgendi
        s2, p2, f1_2 = aami_ec57_metrics(true_r, detect_elgendi(ecg))
        el_se.append(s2); el_ppv.append(p2); el_f1.append(f1_2)

        # 3. Our Quadratic Adaptive (exact C logic mirrored + verified)
        b_c, a_c = butter(3, [5.0/50.0, 15.0/50.0], btype="band")
        y_c = lfilter(b_c, a_c, ecg)
        e_c = np.maximum(0.0, y_c)**2
        det_c = []
        spki, npki, thr, last_p = 0.22, 0.01, 0.094, 0
        for n in range(2, len(e_c)):
            if e_c[n] < e_c[n-1] and e_c[n-1] >= e_c[n-2]:
                c = n - 1
                ep = e_c[n-1]
                if (last_p == 0 or (c - last_p) >= 48) and ep > thr:
                    spki = 0.125 * ep + 0.875 * spki
                    thr = max(0.014, npki + 0.40 * (spki - npki))
                    last_p = c
                    det_c.append(c)
                elif ep > 0.0005:
                    npki = 0.125 * ep + 0.875 * npki
                    thr = max(0.014, npki + 0.40 * (spki - npki))
        s3, p3, f1_3 = aami_ec57_metrics(true_r, np.array(det_c))
        qr_se.append(s3); qr_ppv.append(p3); qr_f1.append(f1_3)

    for name, se_l, ppv_l, f1_l in [
        ("Pan-Tompkins (1985)", pt_se, pt_ppv, pt_f1),
        ("Elgendi Two-MA (2013)", el_se, el_ppv, el_f1),
        ("Proposed Quadratic Adaptive", qr_se, qr_ppv, qr_f1)
    ]:
        baseline_rows.append({
            "Condition": cond,
            "Algorithm": name,
            "AAMI_Sensitivity_Pct": f"{np.mean(se_l):.2f} ± {np.std(se_l, ddof=1):.2f}",
            "AAMI_PPV_Pct": f"{np.mean(ppv_l):.2f} ± {np.std(ppv_l, ddof=1):.2f}",
            "Beat_F1_Pct": f"{np.mean(f1_l):.2f} ± {np.std(f1_l, ddof=1):.2f}"
        })

base_comp_df = pd.DataFrame(baseline_rows)
base_comp_df.to_csv(ROOT / "results/reviewer_aami_baselines.csv", index=False)
print("\n--- AAMI EC57 BEAT-LEVEL BASELINE COMPARISON (10 MIT-BIH PATIENTS) ---")
print(base_comp_df.to_string(index=False))

# 3. WILCOXON SIGNIFICANCE TEST ON CLINICAL MIT-BIH RECOVERY
clin_df = pd.read_csv(ROOT / "results/qemu_40_clinical_trials.csv")
trip = clin_df[clin_df["Scenario"] == "triple_compound_clinical"]
stat, p_val = wilcoxon(trip["Raw_HR_MAE"], trip["Mitigated_HR_MAE"])
print(f"\n--- WILCOXON SIGNED-RANK TEST (TRIPLE COMPOUND CLINICAL MAE, N=10) ---")
print(f"Raw Mean={trip['Raw_HR_MAE'].mean():.3f} (Median={trip['Raw_HR_MAE'].median():.3f}) BPM vs "
      f"Mitigated Mean={trip['Mitigated_HR_MAE'].mean():.3f} (Median={trip['Mitigated_HR_MAE'].median():.3f}) BPM | W={stat}, p={p_val:.4f}")

# 4. TARGET-ONLY CAUSAL VS FILTFILT DISCREPANCY & MEMORY/STACK PROFILING
b_d, a_d = butter(3, [5.0/50.0, 15.0/50.0], btype="band")
zero_phase = filtfilt(b_d, a_d, base_ecg)
causal_c = lfilter(b_d, a_d, base_ecg)
startup_diff = np.max(np.abs(zero_phase[:150] - causal_c[:150]))
phase_lag_samples = np.argmax(np.correlate(zero_phase[200:500], causal_c[200:500], mode="full")) - 299
print(f"\n--- HOST (FILTFILT) VS TARGET (CAUSAL IIR) DISCREPANCY ---")
print(f"Max Start-up Transient Mismatch (first 1.5s): {startup_diff:.4f} mV | Causal Group Delay: {abs(phase_lag_samples)*10} ms")

size_out = subprocess.run(["arm-none-eabi-size", "-A", str(ELF)], capture_output=True, text=True).stdout
print("\n--- ELF SECTION & SRAM FOOTPRINT ---")
print(size_out)
