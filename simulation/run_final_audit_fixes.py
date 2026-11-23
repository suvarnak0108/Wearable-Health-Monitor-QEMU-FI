import numpy as np
import pandas as pd
import subprocess
import re
import struct
from pathlib import Path
from scipy.signal import resample, butter, lfilter
from scipy.stats import wilcoxon
import wfdb

ROOT = Path.cwd()
CACHE_DIR = ROOT / "data_cache"
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"
NATIVE_EXE = ROOT / "firmware/native_x86_check"

C_SRCS = [
    "firmware/src/main.c", "firmware/src/ecg_filter.c",
    "firmware/src/r_peak.c", "firmware/src/hr_estimator.c",
    "firmware/src/fault_detector.c"
]

def load_cached(rec_name, pn_dir, ann=False):
    ext = "atr" if ann else "dat"
    cfile = CACHE_DIR / f"{pn_dir}_{rec_name}_{ext}.npz"
    d = np.load(cfile, allow_pickle=True)
    return (d["sample"], d["symbol"]) if ann else d["sig"]

def compile_arm():
    subprocess.run([
        "arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb",
        "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-ffp-contract=off",
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

def run_qemu(ecg, hr, spo2, dt_ms, labels):
    write_header(ecg, hr, spo2, dt_ms, labels)
    compile_arm()
    cmd = ["qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4",
           "-kernel", str(ELF), "-nographic", "-semihosting-config", "enable=on,target=native"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    out = res.stdout + res.stderr
    def grab(pat):
        m = re.search(pat, out)
        return int(m.group(1)) if m else 0
    hr_raw = np.array([int(m)/100.0 for m in re.findall(r"HR_RAW x100 = (\d+)", out)])
    hr_rr  = np.array([int(m)/100.0 for m in re.findall(r"HR_RR  x100 = (\d+)", out)])
    hr_mit = np.array([int(m)/100.0 for m in re.findall(r"HR_MIT x100 = (\d+)", out)])
    sqi    = np.array([int(m)/100.0 for m in re.findall(r"SQI    x100 = (\d+)", out)])
    m_csv = re.search(r"PEAKS_CSV:\s*([0-9,]+)", out)
    peak_idx = np.array([int(x) for x in m_csv.group(1).split(",") if x]) if m_csv else np.array([], dtype=int)
    rpeaks = grab(r"Detected R-peaks:\s*(\d+)")
    holds  = grab(r"HoldWindows:\s*(\d+)")
    tp, fp, fn, tn = grab(r"TP:\s*(\d+)"), grab(r"FP:\s*(\d+)"), grab(r"FN:\s*(\d+)"), grab(r"TN:\s*(\d+)")
    hwm = grab(r"StackHWM:\s*(\d+)")
    return rpeaks, peak_idx, holds, tp, fp, fn, tn, hwm, hr_raw, hr_rr, hr_mit, sqi

def bootstrap_ci(arr, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def make_realistic_aux(true_hr_100hz, seed=0):
    rng = np.random.default_rng(seed)
    aux_hr = np.zeros(6000); spo2 = np.zeros(6000)
    cur_hr = float(np.round(true_hr_100hz[0] + rng.normal(0, 1.5)))
    cur_spo2 = 98.0
    for n in range(6000):
        if n % 100 == 0:
            lag_idx = max(0, n - 150)
            cur_hr = float(np.round(true_hr_100hz[lag_idx] + rng.normal(0, 1.8)))
            cur_spo2 = float(int(np.clip(np.round(98.0 + rng.normal(0, 0.4)), 96, 99)))
        aux_hr[n] = cur_hr; spo2[n] = cur_spo2
    return aux_hr, spo2

def make_step_ecg():
    fs = 100.0
    regimes = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    true_hr = np.repeat(regimes, 1000)
    ecg = np.zeros(6000); t_cursor = 0.0
    for w, hr_bpm in enumerate(regimes):
        rr = 60.0 / hr_bpm
        while t_cursor < (w + 1) * 10.0:
            dt = np.arange(w*1000, (w+1)*1000)/fs - t_cursor
            ecg[w*1000:(w+1)*1000] += (
                0.12*np.exp(-((dt+0.16)/0.025)**2) - 0.15*np.exp(-((dt+0.035)/0.010)**2)
                + 1.00*np.exp(-(dt/0.012)**2) - 0.22*np.exp(-((dt-0.035)/0.010)**2)
                + 0.28*np.exp(-((dt-0.22)/0.045)**2)
            )
            t_cursor += rr
    return ecg, true_hr, np.array(regimes)

def make_continuous_ecg(seed=0):
    fs = 100.0; t = np.arange(6000) / fs
    true_hr = 95.0 + 35.0 * np.sin(2 * np.pi * t / 60.0) + 5.0 * np.cos(2 * np.pi * t / 17.0)
    phase = np.cumsum(true_hr / 60.0 / fs)
    beat_times = [int(np.searchsorted(phase, k))/fs for k in range(1, int(np.floor(phase[-1])) + 1)
                  if int(np.searchsorted(phase, k)) < 6000]
    ecg = np.zeros(6000)
    for bt in beat_times:
        dt = t - bt
        ecg += (0.12*np.exp(-((dt+0.16)/0.025)**2) - 0.15*np.exp(-((dt+0.035)/0.010)**2)
                + 1.00*np.exp(-(dt/0.012)**2) - 0.22*np.exp(-((dt-0.035)/0.010)**2)
                + 0.28*np.exp(-((dt-0.22)/0.045)**2))
    aux_hr, spo2 = make_realistic_aux(true_hr, seed=seed)
    win_true = np.array([np.mean(true_hr[w*1000:(w+1)*1000]) for w in range(6)])
    return ecg, true_hr, aux_hr, spo2, win_true, len(beat_times)

print("=== A. CONTINUOUS HR EXACT RANGE ===")
_, thr_c, _, _, _, _ = make_continuous_ecg(0)
print(f"Exact Continuous HR min={thr_c.min():.2f} BPM, max={thr_c.max():.2f} BPM")

print("\n=== B. PER-RECORD AAMI EC57 BREAKDOWN (TABLE 7B) ===")
records = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
beat_syms = {"N", "L", "R", "A", "V", "F", "e", "j", "E", "J"}
nstdb_em = resample(load_cached("em", "nstdb"), 6000); nstdb_em /= np.std(nstdb_em)

def aami_score(true_r, det_r, tol=15):
    matched_t = set(); tp = 0
    for d in det_r:
        diffs = np.abs(true_r - d)
        if len(diffs) > 0:
            j = int(np.argmin(diffs))
            if diffs[j] <= tol and j not in matched_t:
                matched_t.add(j); tp += 1
    se = tp / len(true_r) * 100.0 if len(true_r) else 0.0
    ppv = tp / len(det_r) * 100.0 if len(det_r) else 0.0
    f1 = 2*se*ppv/(se+ppv) if (se+ppv) else 0.0
    return tp, len(true_r), len(det_r), se, ppv, f1

for idx_r, rec_id in enumerate(records):
    sig_raw = load_cached(rec_id, "mitdb")
    ann_s, ann_sym = load_cached(rec_id, "mitdb", ann=True)
    ecg = resample(sig_raw, 6000)
    ecg = (ecg - np.median(ecg)) / np.percentile(np.abs(ecg - np.median(ecg)), 99)
    true_r = np.array([int(round(s*100.0/360.0)) for s, sym in zip(ann_s, ann_sym) if sym in beat_syms])
    true_r = true_r[(true_r > 0) & (true_r < 6000)]
    _, prop_peaks, _, _, _, _, _, _, _, _, _, _ = run_qemu(ecg, np.full(6000, 75.0), np.full(6000, 98.0),
                                                            np.full(6000, 10, dtype=np.uint32), np.zeros(6000, dtype=np.uint8))
    tp, n_tr, n_dt, se, ppv, f1 = aami_score(true_r, prop_peaks - 10)
    print(f"Rec {rec_id} | True={n_tr:2d} Det={n_dt:2d} Matched={tp:2d} | Se={se:6.2f}% PPV={ppv:6.2f}% F1={f1:6.2f}%")

print("\n=== C. FIXED F5 (165-SAMPLE FREEZE -> 149 WARM-UP FN + 16 TP) & TABLE 10 (PREC/REC/F1) ===")
ecg_step, thr_step, w_step = make_step_ecg()
# F5 with 165-sample freeze
tp_l, fp_l, fn_l, f1_l, fpr_l = [], [], [], [], []
for tr in range(20):
    aux_s, spo2_s = make_realistic_aux(thr_step, seed=200 + tr)
    dt_s = np.full(6000, 10, dtype=np.uint32); gt = np.zeros(6000, dtype=np.uint8)
    ecg_run = ecg_step.copy()
    s = 2000 + (tr * 37) % 1500
    ecg_run[s:s+165] = ecg_run[s-1]
    gt[s:s+165] = 1
    _, _, _, tp, fp, fn, tn, _, _, _, _, _ = run_qemu(ecg_run, aux_s, spo2_s, dt_s, gt)
    pr = tp/(tp+fp)*100 if (tp+fp) else 100.0
    rc = tp/(tp+fn)*100 if (tp+fn) else 100.0
    f1_l.append(2*pr*rc/(pr+rc) if (pr+rc) else 100.0)
    fpr_l.append(fp/(fp+tn)*100)
    tp_l.append(tp); fp_l.append(fp); fn_l.append(fn)
print(f"F5 Stale (165-sample freeze) | TP={np.mean(tp_l):.1f} FP={np.mean(fp_l):.1f} FN={np.mean(fn_l):.1f} | "
      f"F1={np.mean(f1_l):.2f}±{np.std(f1_l,ddof=1):.2f}% | FPR={np.mean(fpr_l):.2f}%")

# Table 10 Density sweep (Recall, Precision, F1)
for fclass in ["Missing", "Range", "Drift"]:
    row_str = []
    for dens in [0.01, 0.025, 0.05, 0.10, 0.20]:
        rc_l, f1_d = [], []
        n_f = int(round(6000 * dens))
        for tr in range(10):
            rng = np.random.default_rng(800 + tr)
            aux_s, spo2_s = make_realistic_aux(thr_step, seed=850 + tr)
            dt_s = np.full(6000, 10, dtype=np.uint32); gt = np.zeros(6000, dtype=np.uint8)
            idx = rng.choice(np.arange(100, 5900), size=n_f, replace=False)
            if fclass == "Missing": dt_s[idx] = 25
            elif fclass == "Range": aux_s[idx] = 245.0
            else: aux_s[idx] += 22.0
            gt[idx] = 1
            _, _, _, tp, fp, fn, tn, _, _, _, _, _ = run_qemu(ecg_step, aux_s, spo2_s, dt_s, gt)
            pr = tp/(tp+fp)*100 if (tp+fp) else 100.0
            rc = tp/(tp+fn)*100 if (tp+fn) else 100.0
            rc_l.append(rc); f1_d.append(2*pr*rc/(pr+rc))
        row_str.append(f"{dens*100:4.1f}%: R={np.mean(rc_l):5.1f}% F1={np.mean(f1_d):5.1f}%")
    print(f"Table 10 {fclass:8s} | " + " | ".join(row_str))

print("\n=== D. UPDATED CLINICAL MIT-BIH WITH INFORMATIVE SQI & HOLD-LAST-HR ENGAGEMENT ===")
for scen_name, n_amp, do_burst, do_drift in [
    ("Clean Clinical Baseline", 0.0, False, False),
    ("Burst Dropout + Stale Freeze", 0.0, True, False),
    ("NSTDB Electrode Motion (0.20 mV) + Drift", 0.20, False, True),
    ("Triple Compound Fault", 0.15, True, True),
]:
    raw_m, rr_m, mit_m, sqi_m, holds_tot = [], [], [], [], 0
    pat_rows = []
    for idx_r, rec_id in enumerate(records):
        sig_raw = load_cached(rec_id, "mitdb")
        ann_s, ann_sym = load_cached(rec_id, "mitdb", ann=True)
        ecg = resample(sig_raw, 6000)
        ecg = (ecg - np.median(ecg)) / np.percentile(np.abs(ecg - np.median(ecg)), 99)
        true_r = np.array([int(round(s*100.0/360.0)) for s, sym in zip(ann_s, ann_sym) if sym in beat_syms])
        true_r = true_r[(true_r > 0) & (true_r < 6000)]
        w_true = np.array([60.0*100.0/np.median(np.diff(true_r[(true_r>=w*1000)&(true_r<(w+1)*1000)]))
                           if len(true_r[(true_r>=w*1000)&(true_r<(w+1)*1000)]) >= 2 else 75.0 for w in range(6)])
        ecg_f = ecg + (n_amp * np.roll(nstdb_em, idx_r * 400) if n_amp > 0 else 0.0)
        aux_f, spo2_f = make_realistic_aux(np.repeat(w_true, 1000), seed=900 + idx_r)
        dt_f = np.full(6000, 10, dtype=np.uint32); gt = np.zeros(6000, dtype=np.uint8)
        if do_burst:
            for b in range(10):
                s = 300 + b * 500
                dt_f[s:s+10] = 60; gt[s:s+10] = 1
        if do_drift:
            aux_f[2000:3500] += 22.0; gt[2000:3500] = 1
        rp, _, holds, tp, fp, fn, tn, _, h_raw, h_rr, h_mit, sqi = run_qemu(ecg_f, aux_f, spo2_f, dt_f, gt)
        holds_tot += holds
        raw_m.append(np.mean(np.abs(h_raw - w_true)))
        rr_m.append(np.mean(np.abs(h_rr - w_true)))
        mit_m.append(np.mean(np.abs(h_mit - w_true)))
        sqi_m.append(np.mean(sqi))
        if scen_name == "Triple Compound Fault":
            h_med_b = np.array([np.median(h_raw[:w+1]) for w in range(6)])
            pat_rows.append({"Rec": rec_id, "TrueB": len(true_r), "DetP": rp,
                             "RawMAE": float(np.mean(np.abs(h_raw - w_true))),
                             "MedBase": float(np.mean(np.abs(h_med_b - w_true))),
                             "RR_MAE": float(np.mean(np.abs(h_rr - w_true))),
                             "MitMAE": float(np.mean(np.abs(h_mit - w_true))),
                             "SQI": float(np.mean(sqi)), "Holds": holds})
    print(f"{scen_name:42s} | Raw={np.mean(raw_m):.3f} RR={np.mean(rr_m):.3f} Mit={np.mean(mit_m):.3f} | SQI={np.mean(sqi_m):.2f}% | Holds={holds_tot}/60")
    if pat_rows:
        df_p = pd.DataFrame(pat_rows)
        print(df_p.to_string(index=False))
        diffs = df_p["RawMAE"] - df_p["MitMAE"]
        nz = diffs[np.abs(diffs) > 1e-6]
        w_st, p_vl = wilcoxon(nz) if len(nz) > 0 else (0.0, 1.0)
        lo_m, hi_m = bootstrap_ci(diffs.values)
        print(f"Summary Triple Compound: Raw={df_p['RawMAE'].mean():.3f} RR={df_p['RR_MAE'].mean():.3f} Mit={df_p['MitMAE'].mean():.3f} | "
              f"SQI Mean={df_p['SQI'].mean():.2f}% | n={len(nz)}, W={w_st}, p={p_vl:.4f}, Boot95%CI=[{lo_m:.3f}, {hi_m:.3f}]")

print("\n=== E. EXACT QEMU EXECUTED INSTRUCTIONS (-d in_asm) & FLOAT MAX ABS/ULP DIFF ===")
write_header(ecg_step, thr_step, np.full(6000, 98.0), np.full(6000, 10, dtype=np.uint32), np.zeros(6000, dtype=np.uint8))
compile_arm()
asm_res = subprocess.run([
    "qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4", "-d", "in_asm",
    "-kernel", str(ELF), "-nographic", "-semihosting-config", "enable=on,target=native"
], capture_output=True, text=True, timeout=20)
insns = len(re.findall(r"^0x[0-9a-f]+:\s+", asm_res.stderr, re.M))
print(f"QEMU Translated Guest Instructions Logged: {insns}")

# Compare exact 6,000 biquad float outputs between host x86 (-ffp-contract=off) and Cortex-M4 (-mfpu=fpv4-sp-d16)
x86_flt = ROOT / "firmware/x86_flt_dump.c"
x86_flt.write_text('''
#include <stdio.h>
#include <stdint.h>
#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"
int main(void) {
    BiquadFilterState f;
    ECG_Filter_Init(&f);
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {
        float y = ECG_Filter_Process(&f, ecg_data[i]);
        uint32_t bits; __builtin_memcpy(&bits, &y, 4);
        printf("%08x\\n", bits);
    }
    return 0;
}
''')
subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-I", "firmware/include",
                str(x86_flt), "firmware/src/ecg_filter.c", "-o", str(NATIVE_EXE)], check=True)
x86_hex = subprocess.run([str(NATIVE_EXE)], capture_output=True, text=True).stdout.strip().splitlines()
x86_vals = np.array([struct.unpack("!f", bytes.fromhex(h))[0] for h in x86_hex], dtype=np.float32)

# Run biquad dump inside QEMU Cortex-M4
m4_flt = ROOT / "firmware/m4_flt_dump.c"
m4_flt.write_text('''
#include <stdint.h>
#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"
static void sh_w0(const char *s) {
    __asm__ volatile ("mov r0, #0x04\\nmov r1, %0\\nbkpt #0xAB\\n" : : "r"(s) : "r0", "r1", "memory");
}
static void sh_exit(void) {
    __asm__ volatile ("mov r0, #0x18\\nldr r1, =0x20026\\nbkpt #0xAB\\n" : : : "r0", "r1", "memory");
    while (1) {}
}
int main(void) {
    BiquadFilterState f;
    ECG_Filter_Init(&f);
    const char *hex = "0123456789abcdef";
    char buf[10]; buf[8] = '\\n'; buf[9] = '\\0';
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {
        float y = ECG_Filter_Process(&f, ecg_data[i]);
        uint32_t bits; __builtin_memcpy(&bits, &y, 4);
        for (int b = 7; b >= 0; b--) { buf[b] = hex[bits & 0xFU]; bits >>= 4; }
        sh_w0(buf);
    }
    sh_exit();
    return 0;
}
''')
M4_FLT_ELF = ROOT / "firmware/m4_flt_dump.elf"
subprocess.run([
    "arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard",
    "-ffp-contract=off", "-ffreestanding", "-nostdlib", "-O2", "-I", "firmware/include",
    "-Wl,-T,firmware/linker/mps2_an386.ld", "firmware/startup/startup.s",
    str(m4_flt), "firmware/src/ecg_filter.c", "-o", str(M4_FLT_ELF)
], check=True)
m4_out = subprocess.run([
    "qemu-system-arm", "-M", "mps2-an386", "-cpu", "cortex-m4", "-kernel", str(M4_FLT_ELF),
    "-nographic", "-semihosting-config", "enable=on,target=native"
], capture_output=True, text=True).stdout.strip().splitlines()
m4_vals = np.array([struct.unpack("!f", bytes.fromhex(h.strip()))[0] for h in m4_out if len(h.strip()) == 8], dtype=np.float32)

max_abs_diff = float(np.max(np.abs(x86_vals - m4_vals)))
x86_u = x86_vals.view(np.uint32); m4_u = m4_vals.view(np.uint32)
max_ulp_diff = int(np.max(np.abs(x86_u.astype(np.int64) - m4_u.astype(np.int64))))
print(f"6,000-Sample Biquad Output Comparison (x86 vs Cortex-M4): Max Abs Diff = {max_abs_diff:.3e} mV | Max ULP Diff = {max_ulp_diff} ULPs")
