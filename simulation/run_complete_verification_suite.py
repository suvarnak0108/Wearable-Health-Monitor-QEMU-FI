import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

# Ensure repo root & simulation dir are importable
ROOT = Path(__file__).resolve().parent.parent
FW_DIR = ROOT / "firmware"
SIM_DIR = ROOT / "simulation"
RES_DIR = ROOT / "results"
OUT_DIR = RES_DIR / "audit_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM_DIR))

CFLAGS_BASE = [
    "-mcpu=cortex-m4",
    "-mthumb",
    "-mfpu=fpv4-sp-d16",
    "-mfloat-abi=hard",
    "-O2",
    "-ffreestanding",
    "-nostdlib",
    "-ffp-contract=off",
    "-fstack-usage",
    "-Wall",
    "-Wextra",
]


def run_cmd(cmd, cwd=ROOT, check=True):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_pipeline():
    """
    Import helpers from existing simulation scripts without executing top-level runs
    if guarded, or load their definitions cleanly.
    """
    candidates = [
        SIM_DIR / "run_final_audit_fixes.py",
        SIM_DIR / "run_remaining_fixes.py",
        SIM_DIR / "run_unified_master_pipeline.py",
    ]
    namespace = {}
    for cand in candidates:
        if cand.exists():
            src = cand.read_text()
            # Strip top-level execution after `if __name__ == "__main__":` or heavy print sections
            cut = re.split(r'\nif __name__\s*==\s*["\']__main__["\']\s*:', src)[0]
            try:
                exec(compile(cut, str(cand), "exec"), namespace)
            except Exception:
                pass
    return namespace


# ==============================================================================
# STEP 0: CLEAN BUILD, ELF/SIZE/NM/.SU SNAPSHOT, STACK PAINTING & SYMBOL AUDIT
# ==============================================================================
def step0_build_and_profile():
    print("\n=== STEP 0: CLEAN BUILD, ELF SNAPSHOT, .SU & STACK PAINTING ===")
    makefile = FW_DIR / "Makefile"
    if makefile.exists():
        run_cmd(["make", "clean"], cwd=FW_DIR, check=False)

    # Remove stale object / stack-usage / elf artifacts before clean build
    for pattern in ("*.o", "*.su", "*.elf", "*.map"):
        for p in FW_DIR.rglob(pattern):
            try:
                p.unlink()
            except OSError:
                pass

    src_dir = FW_DIR / "src"
    c_sources = sorted([
        q for q in src_dir.glob("*.c")
        if "qemu_test" not in q.name and "pan_tompkins" not in q.name
    ])
    asm_sources = sorted((FW_DIR / "startup").glob("*.s")) + sorted((FW_DIR / "startup").glob("*.S"))
    ld_scripts = sorted(FW_DIR.rglob("*.ld"))

    if not c_sources:
        raise FileNotFoundError(f"No .c firmware sources found under {FW_DIR}")

    obj_files = []
    for src in c_sources:
        obj = src.with_suffix(".o")
        cmd = ["arm-none-eabi-gcc", *CFLAGS_BASE, f"-I{FW_DIR}", f"-I{FW_DIR / 'include'}", "-c", str(src), "-o", str(obj)]
        run_cmd(cmd, cwd=FW_DIR)
        obj_files.append(obj)

    for asm in asm_sources:
        obj = asm.with_suffix(".o")
        cmd = ["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-g", "-c", str(asm), "-o", str(obj)]
        run_cmd(cmd, cwd=FW_DIR)
        obj_files.append(obj)

    elf_path = FW_DIR / "audit_verified_m4.elf"
    link_cmd = ["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-nostdlib", "-Wl,-Map=" + str(FW_DIR / "audit_verified_m4.map")]
    if ld_scripts:
        link_cmd += ["-T", str(ld_scripts[0])]
    link_cmd += [str(o) for o in obj_files] + ["-o", str(elf_path), "-lgcc", "-lc", "-lm"]
    # Fallback without -lc if freestanding toolchain lacks libc
    res = run_cmd(link_cmd, cwd=FW_DIR, check=False)
    if res.returncode != 0:
        link_cmd_bare = [x for x in link_cmd if x != "-lc"]
        run_cmd(link_cmd_bare, cwd=FW_DIR, check=True)

    elf_hash = sha256_file(elf_path)
    size_out = run_cmd(["arm-none-eabi-size", "-A", str(elf_path)], cwd=FW_DIR).stdout
    nm_out = run_cmd(["arm-none-eabi-nm", "-S", "--size-sort", str(elf_path)], cwd=FW_DIR).stdout

    su_records = []
    su_text_all = []
    for su in sorted(FW_DIR.rglob("*.su")):
        txt = su.read_text().strip()
        su_text_all.append(f"--- {su.relative_to(ROOT)} ---\n{txt}")
        for line in txt.splitlines():
            # Format: <file>:<line>:<col>:<func_name>	<bytes>	<static|dynamic|bounded>
            c_parts = line.split(":")
            if len(c_parts) >= 4:
                tail_fields = ":".join(c_parts[3:]).split()
                if len(tail_fields) >= 2 and tail_fields[1].isdigit():
                    func_name = tail_fields[0]
                    stack_bytes = int(tail_fields[1])
                    qualifier = tail_fields[2] if len(tail_fields) > 2 else ""
                su_records.append({
                    "file": str(su.relative_to(ROOT)),
                    "function": func_name,
                    "stack_bytes": stack_bytes,
                    "qualifier": qualifier,
                })

    # Parse symbol sizes from nm -S --size-sort
    symbols = []
    for line in nm_out.splitlines():
        m = re.match(r"^\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([tTdDbBrR])\s+(.+)$", line)
        if m:
            symbols.append({
                "addr": m.group(1),
                "size_bytes": int(m.group(2), 16),
                "type": m.group(3),
                "name": m.group(4).strip(),
            })

    # Run QEMU semihosting to capture output & stack painting telemetry if available
    qemu_proc = run_cmd(
        [
            "qemu-system-arm",
            "-M",
            "mps2-an386",
            "-cpu",
            "cortex-m4",
            "-kernel",
            str(elf_path),
            "-nographic",
            "-semihosting-config",
            "enable=on,target=native",
        ],
        cwd=FW_DIR,
        check=False,
    )
    qemu_log = qemu_proc.stdout + "\n" + qemu_proc.stderr

    # Save snapshots
    (OUT_DIR / "elf_sha256.txt").write_text(f"{elf_hash}  {elf_path.relative_to(ROOT)}\n")
    (OUT_DIR / "arm_none_eabi_size_A.txt").write_text(size_out)
    (OUT_DIR / "arm_none_eabi_nm_size_sort.txt").write_text(nm_out)
    (OUT_DIR / "stack_usage_su.txt").write_text("\n\n".join(su_text_all) + "\n")
    (OUT_DIR / "qemu_stack_paint_run.log").write_text(qemu_log)

    sym_by_name = {s["name"]: s["size_bytes"] for s in symbols}
    su_by_func = {r["function"]: r["stack_bytes"] for r in su_records}

    # Extract painted stack HWM from QEMU log if printed
    hwm_matches = re.findall(r"(?:hwm|high[-_\s]*water|stack)[^0-9\n]*([0-9]{3,5})\s*(?:B|bytes)?", qemu_log, re.I)

    step0_summary = {
        "elf_path": str(elf_path.relative_to(ROOT)),
        "sha256": elf_hash,
        "cflags": CFLAGS_BASE,
        "symbols_bytes": sym_by_name,
        "su_stack_bytes": su_by_func,
        "qemu_returncode": qemu_proc.returncode,
        "detected_hwm_tokens": hwm_matches,
    }
    (OUT_DIR / "step0_build_summary.json").write_text(json.dumps(step0_summary, indent=2))
    print(f"ELF SHA-256: {elf_hash}")
    print("Parsed .su stack bytes:", su_by_func)
    return step0_summary


# ==============================================================================
# SYNTHETIC ECG + 1 Hz AUX + FAULT INJECTION ORACLE (STEPS 1, 2, 3, 4, 5, 7)
# ==============================================================================
def synth_gaussian_ecg(t_vec: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """Section 3 Eq. (1) 5-wave Gaussian ECG synthesizer."""
    amp = [0.12, -0.15, 1.00, -0.22, 0.28]
    off = [-0.16, -0.035, 0.0, 0.035, 0.22]
    wid = [0.018, 0.007, 0.0085, 0.007, 0.032]
    sig = np.zeros_like(t_vec, dtype=np.float64)
    for tk in beat_times:
        for a, d, w in zip(amp, off, wid):
            sig += a * np.exp(-0.5 * ((t_vec - tk - d) / w) ** 2)
    return sig


def hr_trajectory_to_beats(hr_100hz: np.ndarray, fs: float = 100.0):
    """Convert a 100 Hz instantaneous HR(t) trajectory (BPM) into beat sample indices and true peaks."""
    n_samples = len(hr_100hz)
    t = np.arange(n_samples, dtype=np.float64) / fs
    # Phase integration: d(phi)/dt = HR(t)/60 (beats per second)
    rr_sec = np.clip(60.0 / np.maximum(hr_100hz, 20.0), 0.25, 2.5)
    beats = []
    cur_t = 0.4
    while cur_t < t[-1] - 0.1:
        idx = int(round(cur_t * fs))
        if 0 <= idx < n_samples:
            beats.append(idx)
        cur_hr = float(hr_100hz[min(n_samples - 1, max(0, idx))])
        cur_t += 60.0 / np.clip(cur_hr, 30.0, 220.0)
    beats_arr = np.array(beats, dtype=np.int32)
    ecg = synth_gaussian_ecg(t, beats_arr / fs)
    return ecg, beats_arr


def make_1hz_aux(true_hr_100hz: np.ndarray, seed: int, lag_samples: int = 150, jitter_sd: float = 1.8):
    """1 Hz quantized auxiliary HR with 1.5 s (150-sample) lag and N(0, 1.8) jitter."""
    rng = np.random.default_rng(seed)
    n = len(true_hr_100hz)
    aux_hr = np.zeros(n, dtype=np.float64)
    spo2 = np.zeros(n, dtype=np.float64)
    cur_hr = float(np.round(true_hr_100hz[0] + rng.normal(0.0, jitter_sd)))
    cur_spo2 = 98.0
    for i in range(n):
        if i % 100 == 0:
            lag_i = max(0, i - lag_samples)
            cur_hr = float(np.round(true_hr_100hz[lag_i] + rng.normal(0.0, jitter_sd)))
            cur_spo2 = float(int(np.clip(np.round(98.0 + rng.normal(0.0, 0.4)), 96, 99)))
        aux_hr[i] = cur_hr
        spo2[i] = cur_spo2
    return aux_hr, spo2


def biquad_bandpass_3stage(x: np.ndarray, fs: float = 100.0):
    """
    Reference causal 3-stage 5-15 Hz 2nd-order Butterworth bandpass filter in float32.
    """
    from scipy.signal import butter, sosfilt
    sos = butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    y = np.zeros_like(x, dtype=np.float32)
    zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
    sos32 = sos.astype(np.float32)
    for n in range(len(x)):
        out_n, zi = sosfilt(sos32, np.array([x[n]], dtype=np.float32), zi=zi)
        y[n] = out_n[0]
    return y, sos32


def detect_rpeaks_adaptive(y3: np.ndarray, r_min: int = 30, spki_init: float = 0.20, floor_min: float = 0.012):
    """
    Algorithm 1 reference implementation with configurable refractory period r_min
    (30 samples = 300 ms at 100 Hz; 48 samples = 480 ms at 100 Hz).
    """
    e = (y3.astype(np.float32)) ** 2
    spki = float(spki_init)
    npki = 0.0
    theta = max(floor_min, npki + 0.40 * (spki - npki))
    p_last = 0
    peaks = []
    for n in range(2, len(e)):
        if e[n] < e[n - 1] and e[n - 1] >= e[n - 2]:
            c = n - 1
            e_peak = float(e[c])
            dn = c - p_last
            if (p_last == 0 or dn >= r_min) and e_peak > theta:
                spki = 0.125 * e_peak + 0.875 * spki
                theta = max(floor_min, npki + 0.40 * (spki - npki))
                p_last = c
                peaks.append(c)
            elif e_peak > 0.0005:
                npki = 0.125 * e_peak + 0.875 * npki
                theta = max(floor_min, npki + 0.40 * (spki - npki))
    return np.array(peaks, dtype=np.int32)


def classify_faults_5stage(dt_ms: np.ndarray, x: np.ndarray, hr_s: np.ndarray, spo2: np.ndarray, hr_ecg_rolling: np.ndarray):
    """
    Algorithm 2 reference implementation returning per-sample fault codes and k_ecg trace.
    Codes: 0=NONE, 1=COMM_BURST, 2=MISSING_SAMPLE, 3=OUT_OF_RANGE, 4=STALE_DATA, 5=HR_INCONSISTENCY
    """
    n_len = len(x)
    codes = np.zeros(n_len, dtype=np.int32)
    k_trace = np.zeros(n_len, dtype=np.int32)
    k_ecg = 0
    x_prev = float(x[0]) if n_len > 0 else 0.0

    for n in range(n_len):
        dt = float(dt_ms[n])
        xn = float(x[n])
        hrs = float(hr_s[n])
        sp = float(spo2[n])
        hecg = float(hr_ecg_rolling[n])

        if dt >= 50.0:
            k_ecg = 0
            k_trace[n] = k_ecg
            codes[n] = 1
            x_prev = xn
            continue
        if dt > 15.0:
            k_ecg = 0
            k_trace[n] = k_ecg
            codes[n] = 2
            x_prev = xn
            continue
        if (hrs < 30.0 or hrs > 200.0) or (sp < 70.0 or sp > 100.0) or np.isnan(hrs) or np.isnan(xn):
            k_trace[n] = k_ecg
            codes[n] = 3
            x_prev = xn
            continue

        if abs(xn - x_prev) < 1e-6:
            k_ecg = min(250, k_ecg + 1)
        else:
            k_ecg = 0
        x_prev = xn
        k_trace[n] = k_ecg

        if k_ecg >= 150:
            codes[n] = 4
            continue
        if hecg > 0.0 and abs(hrs - hecg) > 12.0:
            codes[n] = 5
            continue
        codes[n] = 0

    return codes, k_trace


def rolling_9beat_hr(peaks: np.ndarray, n_len: int, fs: float = 100.0) -> np.ndarray:
    out = np.zeros(n_len, dtype=np.float64)
    if len(peaks) < 2:
        return out
    rr_inst = np.diff(peaks) / fs  # seconds
    hr_inst = 60.0 / np.maximum(rr_inst, 1e-6)
    p_idx = 0
    for n in range(n_len):
        while p_idx + 1 < len(peaks) and peaks[p_idx + 1] <= n:
            p_idx += 1
        if p_idx >= 1:
            win = hr_inst[max(0, p_idx - 8) : p_idx]
            if len(win) > 0:
                out[n] = float(np.median(win))
    return out


def windowed_10s_hr_rr_filtered(peaks: np.ndarray, n_len: int = 6000, win_size: int = 1000, fs: float = 100.0) -> np.ndarray:
    """10-second (1,000-sample) window median HR with physiological RR bounds [0.3*fs, 1.5*fs] (40..200 BPM)."""
    n_wins = n_len // win_size
    hr_w = np.zeros(n_wins, dtype=np.float64)
    last_hr = 75.0
    for w in range(n_wins):
        s0, s1 = w * win_size, (w + 1) * win_size
        w_peaks = peaks[(peaks >= s0) & (peaks < s1)]
        if len(w_peaks) >= 2:
            rr = np.diff(w_peaks).astype(np.float64)
            valid = rr[(rr >= 0.30 * fs) & (rr <= 1.50 * fs)]
            if len(valid) > 0:
                med_rr = float(np.median(valid))
                last_hr = (60.0 * fs) / med_rr
        hr_w[w] = last_hr
    return hr_w


def eval_sample_metrics(pred_pos: np.ndarray, gt_pos: np.ndarray):
    tp = int(np.sum(pred_pos & gt_pos))
    fp = int(np.sum(pred_pos & (~gt_pos)))
    fn = int(np.sum((~pred_pos) & gt_pos))
    tn = int(np.sum((~pred_pos) & (~gt_pos)))
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else np.nan
    rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else np.nan
    f1 = (2.0 * prec * rec / (prec + rec)) if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else (0.0 if (tp + fn) > 0 else np.nan)
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "prec": prec, "rec": rec, "f1": f1, "fpr": fpr}


# ==============================================================================
# STEP 1: SIGMA = 0 STEP-HR ROW (20 TRIALS)
# ==============================================================================
def step1_sigma_zero_row():
    print("\n=== STEP 1: SIGMA = 0 STEP-HR BASELINE (20 TRIALS) ===")
    t = np.arange(6000, dtype=np.float64) / 100.0
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_100hz = np.repeat(step_vals, 1000)
    ecg_clean, true_beats = hr_trajectory_to_beats(hr_100hz, fs=100.0)
    # True 10s window HR from true beats
    w_true = windowed_10s_hr_rr_filtered(true_beats, 6000, 1000, 100.0)

    rows = []
    for tr in range(20):
        aux_hr, spo2 = make_1hz_aux(hr_100hz, seed=50 + tr)
        dt_ms = np.full(6000, 10, dtype=np.int32)
        gt = np.zeros(6000, dtype=bool)  # zero injected sensor/timing faults
        y3, _ = biquad_bandpass_3stage(ecg_clean, fs=100.0)
        peaks = detect_rpeaks_adaptive(y3, r_min=30)
        hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
        codes, _ = classify_faults_5stage(dt_ms, ecg_clean, aux_hr, spo2, hr_roll)
        m = eval_sample_metrics(codes != 0, gt)
        h_rr = windowed_10s_hr_rr_filtered(peaks, 6000, 1000, 100.0)
        mae = float(np.mean(np.abs(h_rr - w_true)))
        rows.append({"trial": tr, "peaks": int(len(peaks)), "fpr": m["fpr"], "fp": m["fp"], "tn": m["tn"], "rr_mae": mae})

    summary = {
        "n_trials": 20,
        "mean_peaks": float(np.mean([r["peaks"] for r in rows])),
        "sd_peaks": float(np.std([r["peaks"] for r in rows], ddof=1)),
        "mean_fpr": float(np.mean([r["fpr"] for r in rows])),
        "sd_fpr": float(np.std([r["fpr"] for r in rows], ddof=1)),
        "mean_rr_mae": float(np.mean([r["rr_mae"] for r in rows])),
        "sd_rr_mae": float(np.std([r["rr_mae"] for r in rows], ddof=1)),
        "trials": rows,
    }
    (OUT_DIR / "step1_sigma0.json").write_text(json.dumps(summary, indent=2))
    print(f"sigma=0 -> Peaks={summary['mean_peaks']:.2f}±{summary['sd_peaks']:.2f}, "
          f"FPR={summary['mean_fpr']:.2f}±{summary['sd_fpr']:.2f}%, "
          f"RR-MAE={summary['mean_rr_mae']:.3f}±{summary['sd_rr_mae']:.3f} BPM")
    return summary


# ==============================================================================
# STEP 2: CONSTANT-HR BOUNDARY PROBE (80 BPM FOR FULL 60s, t in [20, 35]s, N=20/OFFSET)
# ==============================================================================
def step2_constant_hr_boundary_probe(n_trials_per_offset: int = 20):
    print("\n=== STEP 2: CONSTANT 80 BPM BOUNDARY PROBE (t in [20, 35]s) ===")
    hr_const = np.full(6000, 80.0, dtype=np.float64)
    ecg_const, peaks_const = hr_trajectory_to_beats(hr_const, fs=100.0)
    y3_const, _ = biquad_bandpass_3stage(ecg_const, fs=100.0)
    peaks_det = detect_rpeaks_adaptive(y3_const, r_min=30)
    hr_roll = rolling_9beat_hr(peaks_det, 6000, fs=100.0)
    dt_ms = np.full(6000, 10, dtype=np.int32)

    w_start, w_end = 2000, 3500  # t in [20, 35] s
    offsets = [10.0, 11.0, 12.0, 13.0, 14.0]
    results = {}

    for off in offsets:
        win_flags_fracs = []
        residuals_mean = []
        residuals_sd = []
        trial_recs = []
        for tr in range(n_trials_per_offset):
            seed = 800 + tr
            aux_hr, spo2 = make_1hz_aux(hr_const, seed=seed)
            aux_mod = aux_hr.copy()
            aux_mod[w_start:w_end] += off
            codes, _ = classify_faults_5stage(dt_ms, ecg_const, aux_mod, spo2, hr_roll)
            flagged = (codes == 5)
            in_win = flagged[w_start:w_end]
            frac = float(np.mean(in_win) * 100.0)
            # Residual HR_s - HR_ecg over window
            valid_ecg = hr_roll[w_start:w_end] > 0
            if np.any(valid_ecg):
                res_vec = aux_mod[w_start:w_end][valid_ecg] - hr_roll[w_start:w_end][valid_ecg]
                r_mean = float(np.mean(res_vec))
                r_sd = float(np.std(res_vec, ddof=1)) if len(res_vec) > 1 else 0.0
            else:
                r_mean, r_sd = float(np.mean(aux_mod[w_start:w_end] - 80.0)), 0.0

            win_flags_fracs.append(frac)
            residuals_mean.append(r_mean)
            residuals_sd.append(r_sd)
            trial_recs.append({
                "trial": tr,
                "seed": seed,
                "in_window_flags": int(np.sum(in_win)),
                "in_window_pct": frac,
                "residual_mean": r_mean,
                "residual_sd": r_sd,
            })

        results[f"+{off:.1f}"] = {
            "offset_bpm": off,
            "n_trials": n_trials_per_offset,
            "flag_pct_mean": float(np.mean(win_flags_fracs)),
            "flag_pct_sd": float(np.std(win_flags_fracs, ddof=1)),
            "residual_bpm_mean": float(np.mean(residuals_mean)),
            "residual_bpm_sd": float(np.mean(residuals_sd)),
            "trials": trial_recs,
        }
        print(f"Offset +{off:.1f} BPM -> Flagged={results[f'+{off:.1f}']['flag_pct_mean']:.1f}% "
              f"± {results[f'+{off:.1f}']['flag_pct_sd']:.1f}% | "
              f"Mean(HR_s - HR_ecg)={results[f'+{off:.1f}']['residual_bpm_mean']:.2f} BPM")

    (OUT_DIR / "step2_constant_boundary_probe.json").write_text(json.dumps(results, indent=2))
    return results


# ==============================================================================
# STEP 3: PER-SAMPLE TRACE FOR F5-ONLY VS BURST + STALE
# ==============================================================================
def step3_stale_trace():
    print("\n=== STEP 3: PER-SAMPLE STALE TRACE (F5-ONLY VS BURST + STALE) ===")
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_100hz = np.repeat(step_vals, 1000)
    ecg_base, _ = hr_trajectory_to_beats(hr_100hz, fs=100.0)
    y3, _ = biquad_bandpass_3stage(ecg_base, fs=100.0)
    peaks = detect_rpeaks_adaptive(y3, r_min=30)
    hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)

    # Trial A: F5-only (165 repeated samples starting at index 3200)
    aux_a, spo2_a = make_1hz_aux(hr_100hz, seed=300)
    dt_a = np.full(6000, 10, dtype=np.int32)
    ecg_a = ecg_base.copy()
    stale_s, stale_e = 3200, 3365
    ecg_a[stale_s:stale_e] = ecg_a[stale_s - 1]
    gt_a = np.zeros(6000, dtype=bool)
    gt_a[stale_s:stale_e] = True
    codes_a, k_a = classify_faults_5stage(dt_a, ecg_a, aux_a, spo2_a, hr_roll)

    # Trial B: Burst + 1.65s Stale (5 bursts of 10 samples + 165-sample freeze)
    aux_b, spo2_b = make_1hz_aux(hr_100hz, seed=300)
    dt_b = np.full(6000, 10, dtype=np.int32)
    ecg_b = ecg_base.copy()
    gt_b = np.zeros(6000, dtype=bool)
    for b in range(5):
        s = 400 + b * 900
        dt_b[s : s + 10] = 60
        gt_b[s : s + 10] = True
    ecg_b[stale_s:stale_e] = ecg_b[stale_s - 1]
    gt_b[stale_s:stale_e] = True
    codes_b, k_b = classify_faults_5stage(dt_b, ecg_b, aux_b, spo2_b, hr_roll)

    code_names = {0: "NONE", 1: "COMM_BURST", 2: "MISSING_SAMPLE", 3: "OUT_OF_RANGE", 4: "STALE_DATA", 5: "HR_INCONSISTENCY"}

    def summarize_window(codes, k_arr, gt, ecg_arr):
        first_stale = next((int(idx) for idx in range(stale_s, stale_e) if codes[idx] == 4), None)
        win_codes = [code_names[int(codes[idx])] for idx in range(stale_s, stale_e)]
        counts = {name: int(win_codes.count(name)) for name in code_names.values()}
        return {
            "first_stale_sample_idx": first_stale,
            "offset_within_165_freeze": (first_stale - stale_s) if first_stale is not None else None,
            "stale_window_code_counts": counts,
            "total_run_tp": int(np.sum((codes != 0) & gt)),
            "total_run_fp": int(np.sum((codes != 0) & (~gt))),
            "total_run_fn": int(np.sum((codes == 0) & gt)),
        }

    trace_a = []
    for idx in range(stale_s - 2, stale_e + 2):
        dx = abs(float(ecg_a[idx]) - float(ecg_a[idx - 1])) if idx > 0 else 0.0
        trace_a.append({
            "n": idx,
            "dt_ms": int(dt_a[idx]),
            "x": float(ecg_a[idx]),
            "abs_dx": dx,
            "k_ecg": int(k_a[idx]),
            "code": code_names[int(codes_a[idx])],
            "gt": bool(gt_a[idx]),
        })

    trace_b = []
    for idx in range(stale_s - 2, stale_e + 2):
        dx = abs(float(ecg_b[idx]) - float(ecg_b[idx - 1])) if idx > 0 else 0.0
        trace_b.append({
            "n": idx,
            "dt_ms": int(dt_b[idx]),
            "x": float(ecg_b[idx]),
            "abs_dx": dx,
            "k_ecg": int(k_b[idx]),
            "code": code_names[int(codes_b[idx])],
            "gt": bool(gt_b[idx]),
        })

    with (OUT_DIR / "step3_f5_only_trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_a[0].keys()))
        w.writeheader()
        w.writerows(trace_a)

    with (OUT_DIR / "step3_burst_stale_trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_b[0].keys()))
        w.writeheader()
        w.writerows(trace_b)

    summary = {
        "f5_only": summarize_window(codes_a, k_a, gt_a, ecg_a),
        "burst_plus_stale": summarize_window(codes_b, k_b, gt_b, ecg_b),
    }
    (OUT_DIR / "step3_stale_trace_summary.json").write_text(json.dumps(summary, indent=2))
    print("F5-only summary:", summary["f5_only"])
    print("Burst+Stale summary:", summary["burst_plus_stale"])
    return summary


# ==============================================================================
# STEP 4: CLINICAL SCENARIO BOOKKEEPING (POOLED VS PER-RECORD MEAN, POSITIVES)
# ==============================================================================
def step4_clinical_bookkeeping():
    print("\n=== STEP 4: CLINICAL SCENARIO BOOKKEEPING (MIT-BIH 10 RECORDS) ===")
    try:
        import wfdb
    except ImportError:
        print("wfdb not installed; skipping MIT-BIH direct re-bookkeeping.")
        return None

    from scipy.signal import resample

    records = ["100", "101", "103", "105", "106", "108", "113", "115", "119", "200"]
    nstdb_em_raw = wfdb.rdsamp("em", pn_dir="nstdb", sampfrom=0, sampto=21600)[0][:, 0]
    nstdb_em = resample(nstdb_em_raw, 6000)
    nstdb_em = (nstdb_em - np.mean(nstdb_em)) / np.std(nstdb_em)

    scenarios = [
        ("Clean Clinical Baseline", 0.0, False, False),
        ("Burst Dropout + Stale Freeze", 0.0, True, False),
        ("NSTDB Electrode Motion (0.20 mV) + Drift", 0.20, False, True),
        ("Triple Compound Fault", 0.15, True, True),
    ]

    book = {}
    for sname, n_amp, do_burst, do_drift in scenarios:
        per_rec = []
        tot_tp = tot_fp = tot_fn = tot_tn = tot_pos = 0
        for r_idx, rec_id in enumerate(records):
            sig = wfdb.rdsamp(rec_id, pn_dir="mitdb", sampfrom=0, sampto=21600)[0][:, 0]
            ecg = resample(sig, 6000)
            med = np.median(ecg)
            scale = np.percentile(np.abs(ecg - med), 99)
            ecg_norm = (ecg - med) / (scale if scale > 1e-9 else 1.0)

            ecg_run = ecg_norm + (n_amp * nstdb_em if n_amp > 0 else 0.0)
            aux_hr, spo2 = make_1hz_aux(np.full(6000, 75.0), seed=900 + r_idx)
            dt_ms = np.full(6000, 10, dtype=np.int32)
            gt = np.zeros(6000, dtype=bool)

            if do_burst:
                for b in range(10):
                    s = 300 + b * 500
                    dt_ms[s : s + 10] = 60
                    gt[s : s + 10] = True
            if do_drift:
                aux_hr[2000:3500] += 22.0
                gt[2000:3500] = True

            y3, _ = biquad_bandpass_3stage(ecg_run, fs=100.0)
            peaks = detect_rpeaks_adaptive(y3, r_min=30)
            hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
            codes, _ = classify_faults_5stage(dt_ms, ecg_run, aux_hr, spo2, hr_roll)
            m = eval_sample_metrics(codes != 0, gt)
            n_pos = int(np.sum(gt))

            tot_tp += m["tp"]
            tot_fp += m["fp"]
            tot_fn += m["fn"]
            tot_tn += m["tn"]
            tot_pos += n_pos
            per_rec.append({"record": rec_id, "positives": n_pos, **m})

        pooled_prec = (tot_tp / (tot_tp + tot_fp) * 100.0) if (tot_tp + tot_fp) > 0 else np.nan
        pooled_rec = (tot_tp / (tot_tp + tot_fn) * 100.0) if (tot_tp + tot_fn) > 0 else np.nan
        pooled_f1 = (2.0 * pooled_prec * pooled_rec / (pooled_prec + pooled_rec)) if (not np.isnan(pooled_prec) and not np.isnan(pooled_rec) and (pooled_prec + pooled_rec) > 0) else np.nan
        pooled_fpr = (tot_fp / (tot_fp + tot_tn) * 100.0) if (tot_fp + tot_tn) > 0 else 0.0

        valid_f1 = [x["f1"] for x in per_rec if not np.isnan(x["f1"])]
        valid_pr = [x["prec"] for x in per_rec if not np.isnan(x["prec"])]
        valid_rc = [x["rec"] for x in per_rec if not np.isnan(x["rec"])]

        book[sname] = {
            "total_positives_10_recs": tot_pos,
            "mean_positives_per_rec": tot_pos / len(records),
            "pooled": {"tp": tot_tp, "fp": tot_fp, "fn": tot_fn, "prec": pooled_prec, "rec": pooled_rec, "f1": pooled_f1, "fpr": pooled_fpr},
            "macro_mean_over_recs": {
                "prec": float(np.mean(valid_pr)) if valid_pr else np.nan,
                "rec": float(np.mean(valid_rc)) if valid_rc else np.nan,
                "f1": float(np.mean(valid_f1)) if valid_f1 else np.nan,
            },
            "per_record": per_rec,
        }
        print(f"{sname}: Pos={tot_pos/10:.1f}/rec | Pooled P/R/F1/FPR = {pooled_prec:.2f}% / {pooled_rec:.2f}% / {pooled_f1:.2f}% / {pooled_fpr:.2f}% | "
              f"Macro F1 = {book[sname]['macro_mean_over_recs']['f1']:.2f}%")

    (OUT_DIR / "step4_clinical_bookkeeping.json").write_text(json.dumps(book, indent=2))
    return book


# ==============================================================================
# STEP 5: REFRACTORY ABLATION (R_min = 48 [480 ms] VS R_min = 30 [300 ms])
# ==============================================================================
def step5_refractory_ablation():
    print("\n=== STEP 5: REFRACTORY ABLATION (480 ms vs 300 ms) ===")
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_step = np.repeat(step_vals, 1000)
    ecg_step, true_beats = hr_trajectory_to_beats(hr_step, fs=100.0)
    w_true = windowed_10s_hr_rr_filtered(true_beats, 6000, 1000, 100.0)

    out = {}
    for label, r_min in [("480ms_Rmin48", 48), ("300ms_Rmin30", 30)]:
        noise_summary = {}
        for sigma in [0.08, 0.15, 0.25, 0.35]:
            peaks_l, fpr_l, mae_l = [], [], []
            for tr in range(20):
                rng = np.random.default_rng(100 + tr)
                ecg_n = ecg_step + rng.normal(0.0, sigma, size=6000)
                aux_hr, spo2 = make_1hz_aux(hr_step, seed=100 + tr)
                dt_ms = np.full(6000, 10, dtype=np.int32)
                gt = np.zeros(6000, dtype=bool)
                y3, _ = biquad_bandpass_3stage(ecg_n, fs=100.0)
                peaks = detect_rpeaks_adaptive(y3, r_min=r_min)
                hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
                codes, _ = classify_faults_5stage(dt_ms, ecg_n, aux_hr, spo2, hr_roll)
                m = eval_sample_metrics(codes != 0, gt)
                h_rr = windowed_10s_hr_rr_filtered(peaks, 6000, 1000, 100.0)
                peaks_l.append(len(peaks))
                fpr_l.append(m["fpr"])
                mae_l.append(float(np.mean(np.abs(h_rr - w_true))))
            noise_summary[f"sigma_{sigma:.2f}"] = {
                "peaks_mean": float(np.mean(peaks_l)),
                "peaks_sd": float(np.std(peaks_l, ddof=1)),
                "fpr_mean": float(np.mean(fpr_l)),
                "fpr_sd": float(np.std(fpr_l, ddof=1)),
                "mae_mean": float(np.mean(mae_l)),
                "mae_sd": float(np.std(mae_l, ddof=1)),
            }
        out[label] = noise_summary
        s25 = noise_summary["sigma_0.25"]
        print(f"{label} @ sigma=0.25 mV -> Peaks={s25['peaks_mean']:.2f}, FPR={s25['fpr_mean']:.2f}%, MAE={s25['mae_mean']:.3f}±{s25['mae_sd']:.3f} BPM")

    (OUT_DIR / "step5_refractory_ablation.json").write_text(json.dumps(out, indent=2))
    return out


# ==============================================================================
# STEP 7: MULTI-SEED CLINICAL REPEATS (5 SEEDS/RECORD) & LATENCY DISTRIBUTIONS
# ==============================================================================
def step7_multiseed_and_latencies():
    print("\n=== STEP 7: LATENCY DISTRIBUTIONS FOR F5, F6, F9 ===")
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_step = np.repeat(step_vals, 1000)
    ecg_step, _ = hr_trajectory_to_beats(hr_step, fs=100.0)
    y3, _ = biquad_bandpass_3stage(ecg_step, fs=100.0)
    peaks = detect_rpeaks_adaptive(y3, r_min=30)
    hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
    dt_ms = np.full(6000, 10, dtype=np.int32)

    lat_f5, lat_f6, lat_f9 = [], [], []
    for tr in range(20):
        aux_base, spo2 = make_1hz_aux(hr_step, seed=400 + tr)

        # F5: freeze starting at n=3200
        ecg_f5 = ecg_step.copy()
        ecg_f5[3200:3365] = ecg_f5[3199]
        c5, _ = classify_faults_5stage(dt_ms, ecg_f5, aux_base, spo2, hr_roll)
        idx5 = next((i for i in range(3200, 3365) if c5[i] == 4), None)
        if idx5 is not None:
            lat_f5.append((idx5 - 3200) * 0.01)

        # F6: +22 BPM drift starting at n=2000
        aux_f6 = aux_base.copy()
        aux_f6[2000:3500] += 22.0
        c6, _ = classify_faults_5stage(dt_ms, ecg_step, aux_f6, spo2, hr_roll)
        idx6 = next((i for i in range(2000, 3500) if c6[i] == 5), None)
        if idx6 is not None:
            lat_f6.append((idx6 - 2000) * 0.01)

        # F9: +1.5 BPM/s ramp starting at t=20s (n=2000) over [20, 38]s
        aux_f9 = aux_base.copy()
        ramp = np.linspace(0.0, 27.0, 1800)
        aux_f9[2000:3800] += ramp
        c9, _ = classify_faults_5stage(dt_ms, ecg_step, aux_f9, spo2, hr_roll)
        idx9 = next((i for i in range(2000, 3800) if c9[i] == 5), None)
        if idx9 is not None:
            lat_f9.append((idx9 - 2000) * 0.01)

    def stats(arr):
        if not arr:
            return {"n": 0}
        a = np.array(arr, dtype=np.float64)
        return {
            "n": int(len(a)),
            "mean_s": float(np.mean(a)),
            "sd_s": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "median_s": float(np.median(a)),
            "min_s": float(np.min(a)),
            "max_s": float(np.max(a)),
        }

    lat_summary = {"F5_stale_s": stats(lat_f5), "F6_step_drift_s": stats(lat_f6), "F9_ramp_1p5_s": stats(lat_f9)}
    (OUT_DIR / "step7_latencies.json").write_text(json.dumps(lat_summary, indent=2))
    print("Latency summary:", lat_summary)
    return lat_summary


# ==============================================================================
# TOOLING VERSIONS SNAPSHOT FOR README / REPRODUCIBILITY
# ==============================================================================
def collect_versions():
    def first_line(cmd):
        p = run_cmd(cmd, check=False)
        out = (p.stdout or p.stderr).strip().splitlines()
        return out[0] if out else "unknown"

    import wfdb
    import scipy
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wfdb": getattr(wfdb, "__version__", "unknown"),
        "arm_none_eabi_gcc": first_line(["arm-none-eabi-gcc", "--version"]),
        "gcc_host": first_line(["gcc", "--version"]),
        "qemu_system_arm": first_line(["qemu-system-arm", "--version"]),
    }
    (OUT_DIR / "environment_versions.json").write_text(json.dumps(versions, indent=2))
    print("Tool versions:", versions)
    return versions


def main():
    collect_versions()
    step0_build_and_profile()
    step1_sigma_zero_row()
    step2_constant_hr_boundary_probe(n_trials_per_offset=20)
    step3_stale_trace()
    step4_clinical_bookkeeping()
    step5_refractory_ablation()
    step7_multiseed_and_latencies()
    print(f"\nAll verification outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
