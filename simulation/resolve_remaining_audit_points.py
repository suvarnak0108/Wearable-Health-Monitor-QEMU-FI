import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import butter, resample, sosfilt

ROOT = Path(__file__).resolve().parent.parent
FW_DIR = ROOT / "firmware"
OUT_DIR = ROOT / "results" / "audit_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFLAGS_M4 = [
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


def synth_gaussian_ecg(t_vec: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    amp = [0.12, -0.15, 1.00, -0.22, 0.28]
    off = [-0.16, -0.035, 0.0, 0.035, 0.22]
    wid = [0.018, 0.007, 0.0085, 0.007, 0.032]
    sig = np.zeros_like(t_vec, dtype=np.float64)
    for tk in beat_times:
        for a, d, w in zip(amp, off, wid):
            sig += a * np.exp(-0.5 * ((t_vec - tk - d) / w) ** 2)
    return sig


def hr_trajectory_to_beats(hr_100hz: np.ndarray, fs: float = 100.0):
    n_samples = len(hr_100hz)
    t = np.arange(n_samples, dtype=np.float64) / fs
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
    rng = np.random.default_rng(seed)
    n = len(true_hr_100hz)
    aux_hr = np.zeros(n, dtype=np.float64)
    spo2 = np.zeros(n, dtype=np.float64)
    cur_hr = float(np.round(true_hr_100hz[0] + (rng.normal(0.0, jitter_sd) if jitter_sd > 0 else 0.0)))
    cur_spo2 = 98.0
    for i in range(n):
        if i % 100 == 0:
            lag_i = max(0, i - lag_samples)
            j = float(rng.normal(0.0, jitter_sd)) if jitter_sd > 0 else 0.0
            cur_hr = float(np.round(true_hr_100hz[lag_i] + j))
            cur_spo2 = float(int(np.clip(np.round(98.0 + (rng.normal(0.0, 0.4) if jitter_sd > 0 else 0.0)), 96, 99)))
        aux_hr[i] = cur_hr
        spo2[i] = cur_spo2
    return aux_hr, spo2


def biquad_bandpass_3stage(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    sos = butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos").astype(np.float32)
    y = sosfilt(sos, np.asarray(x, dtype=np.float32))
    return np.asarray(y, dtype=np.float32)


def detect_rpeaks_adaptive(y3: np.ndarray, r_min: int = 30, spki_init: float = 0.20, floor_min: float = 0.012):
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


def rolling_9beat_hr(peaks: np.ndarray, n_len: int, fs: float = 100.0) -> np.ndarray:
    out = np.zeros(n_len, dtype=np.float64)
    if len(peaks) < 2:
        return out
    hr_inst = 60.0 / np.maximum(np.diff(peaks) / fs, 1e-6)
    p_idx = 0
    for n in range(n_len):
        while p_idx + 1 < len(peaks) and peaks[p_idx + 1] <= n:
            p_idx += 1
        if p_idx >= 1:
            win = hr_inst[max(0, p_idx - 8) : p_idx]
            if len(win) > 0:
                out[n] = float(np.median(win))
    return out


def classify_faults_5stage(dt_ms, x, hr_s, spo2, hr_ecg_rolling):
    n_len = len(x)
    codes = np.zeros(n_len, dtype=np.int32)
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
            codes[n] = 1
            x_prev = xn
            continue
        if dt > 15.0:
            k_ecg = 0
            codes[n] = 2
            x_prev = xn
            continue
        if (hrs < 30.0 or hrs > 200.0) or (sp < 70.0 or sp > 100.0) or np.isnan(hrs) or np.isnan(xn):
            codes[n] = 3
            x_prev = xn
            continue
        if abs(xn - x_prev) < 1e-6:
            k_ecg = min(250, k_ecg + 1)
        else:
            k_ecg = 0
        x_prev = xn
        if k_ecg >= 150:
            codes[n] = 4
            continue
        if hecg > 0.0 and abs(hrs - hecg) > 12.0:
            codes[n] = 5
            continue
        codes[n] = 0
    return codes


def windowed_10s_hr_rr_filtered(peaks: np.ndarray, n_len: int = 6000, win_size: int = 1000, fs: float = 100.0) -> np.ndarray:
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
                last_hr = (60.0 * fs) / float(np.median(valid))
        hr_w[w] = last_hr
    return hr_w


def build_clean_elf():
    for p in FW_DIR.rglob("*.o"):
        p.unlink(missing_ok=True)
    for p in FW_DIR.rglob("*.su"):
        p.unlink(missing_ok=True)
    c_srcs = sorted([q for q in (FW_DIR / "src").glob("*.c") if "qemu_test" not in q.name and "pan_tompkins" not in q.name])
    asm_srcs = sorted((FW_DIR / "startup").glob("*.s")) + sorted((FW_DIR / "startup").glob("*.S"))
    ld_scripts = sorted((FW_DIR / "linker").glob("*.ld"))

    objs = []
    for s in c_srcs:
        o = s.with_suffix(".o")
        run_cmd(["arm-none-eabi-gcc", *CFLAGS_M4, f"-I{FW_DIR}", f"-I{FW_DIR / 'include'}", "-c", str(s), "-o", str(o)], cwd=FW_DIR)
        objs.append(o)
    for a in asm_srcs:
        o = a.with_suffix(".o")
        run_cmd(["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-g", "-c", str(a), "-o", str(o)], cwd=FW_DIR)
        objs.append(o)

    elf_path = FW_DIR / "firmware_m4.elf"
    link_cmd = ["arm-none-eabi-gcc", "-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-nostdlib"]
    if ld_scripts:
        link_cmd += ["-T", str(ld_scripts[0])]
    link_cmd += [str(o) for o in objs] + ["-o", str(elf_path), "-lgcc", "-lm"]
    run_cmd(link_cmd, cwd=FW_DIR)

    su_map = {}
    for su in FW_DIR.rglob("*.su"):
        for line in su.read_text().splitlines():
            parts = line.split(":")
            if len(parts) >= 4:
                tail = ":".join(parts[3:]).split()
                if len(tail) >= 2 and tail[1].isdigit():
                    su_map[tail[0]] = int(tail[1])

    summary = {"elf": str(elf_path.relative_to(ROOT)), "sha256": sha256_file(elf_path), "su_bytes": su_map}
    (OUT_DIR / "elf_build.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_unified_noise_and_zero_lag_check():
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_step = np.repeat(step_vals, 1000)
    ecg_step, true_beats = hr_trajectory_to_beats(hr_step, fs=100.0)
    w_true = windowed_10s_hr_rr_filtered(true_beats, 6000, 1000, 100.0)
    dt_ms = np.full(6000, 10, dtype=np.int32)
    gt = np.zeros(6000, dtype=bool)

    sigmas = [0.0, 0.08, 0.15, 0.25, 0.35]
    report = {"300ms_Rmin30": {}, "480ms_Rmin48": {}, "zero_lag_sigma_0.08": {}}

    for label, r_min in [("300ms_Rmin30", 30), ("480ms_Rmin48", 48)]:
        for sigma in sigmas:
            peaks_l, fpr_l, mae_l = [], [], []
            for tr in range(20):
                seed = 1000 + tr
                rng = np.random.default_rng(seed)
                ecg_n = ecg_step + (rng.normal(0.0, sigma, size=6000) if sigma > 0 else 0.0)
                aux_hr, spo2 = make_1hz_aux(hr_step, seed=seed, lag_samples=150, jitter_sd=1.8)
                y3 = biquad_bandpass_3stage(ecg_n, fs=100.0)
                peaks = detect_rpeaks_adaptive(y3, r_min=r_min)
                hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
                codes = classify_faults_5stage(dt_ms, ecg_n, aux_hr, spo2, hr_roll)
                fp = int(np.sum((codes != 0) & (~gt)))
                tn = int(np.sum((codes == 0) & (~gt)))
                fpr_l.append(fp / (fp + tn) * 100.0)
                h_rr = windowed_10s_hr_rr_filtered(peaks, 6000, 1000, 100.0)
                mae_l.append(float(np.mean(np.abs(h_rr - w_true))))
                peaks_l.append(len(peaks))
            report[label][f"sigma_{sigma:.2f}"] = {
                "peaks_mean": float(np.mean(peaks_l)),
                "peaks_sd": float(np.std(peaks_l, ddof=1)),
                "fpr_mean": float(np.mean(fpr_l)),
                "fpr_sd": float(np.std(fpr_l, ddof=1)),
                "mae_mean": float(np.mean(mae_l)),
                "mae_sd": float(np.std(mae_l, ddof=1)),
            }

    # Zero-lag test at sigma=0.08
    fpr_zl = []
    for tr in range(20):
        seed = 1000 + tr
        rng = np.random.default_rng(seed)
        ecg_n = ecg_step + rng.normal(0.0, 0.08, size=6000)
        aux_hr, spo2 = make_1hz_aux(hr_step, seed=seed, lag_samples=0, jitter_sd=0.0)
        y3 = biquad_bandpass_3stage(ecg_n, fs=100.0)
        peaks = detect_rpeaks_adaptive(y3, r_min=30)
        hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
        codes = classify_faults_5stage(dt_ms, ecg_n, aux_hr, spo2, hr_roll)
        fp = int(np.sum((codes != 0) & (~gt)))
        tn = int(np.sum((codes == 0) & (~gt)))
        fpr_zl.append(fp / (fp + tn) * 100.0)
    report["zero_lag_sigma_0.08"] = {"fpr_mean": float(np.mean(fpr_zl)), "fpr_sd": float(np.std(fpr_zl, ddof=1))}

    (OUT_DIR / "unified_noise_and_refractory.json").write_text(json.dumps(report, indent=2))
    return report


def run_randomized_stale_and_compound():
    step_vals = [70.0, 84.0, 75.0, 95.0, 80.0, 70.0]
    hr_step = np.repeat(step_vals, 1000)
    ecg_base, _ = hr_trajectory_to_beats(hr_step, fs=100.0)
    y3 = biquad_bandpass_3stage(ecg_base, fs=100.0)
    peaks = detect_rpeaks_adaptive(y3, r_min=30)
    hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
    dt_base = np.full(6000, 10, dtype=np.int32)

    valid_starts = [s for s in range(200, 5800 - 165) if not any(b <= s < b + 200 for b in [1000, 2000, 3000, 4000, 5000])]
    stale_only_rec, stale_only_overall_rec = [], []
    comp_stale_rec, comp_overall_rec, comp_f1 = [], [], []

    for tr in range(20):
        rng = np.random.default_rng(2000 + tr)
        s_idx = int(rng.choice(valid_starts))
        e_idx = s_idx + 165
        aux_hr, spo2 = make_1hz_aux(hr_step, seed=2000 + tr)

        ecg_f5 = ecg_base.copy()
        ecg_f5[s_idx:e_idx] = ecg_f5[s_idx - 1]
        gt_f5 = np.zeros(6000, dtype=bool)
        gt_f5[s_idx:e_idx] = True
        c_f5 = classify_faults_5stage(dt_base, ecg_f5, aux_hr, spo2, hr_roll)
        stale_only_rec.append(float(np.mean(c_f5[s_idx:e_idx] == 4) * 100.0))
        tp5 = int(np.sum((c_f5 != 0) & gt_f5))
        fn5 = int(np.sum((c_f5 == 0) & gt_f5))
        stale_only_overall_rec.append(tp5 / (tp5 + fn5) * 100.0)

        ecg_c = ecg_base.copy()
        dt_c = dt_base.copy()
        gt_c = np.zeros(6000, dtype=bool)
        for b in range(5):
            bs = 400 + b * 900
            dt_c[bs : bs + 10] = 60
            gt_c[bs : bs + 10] = True
        ecg_c[s_idx:e_idx] = ecg_c[s_idx - 1]
        gt_c[s_idx:e_idx] = True
        c_c = classify_faults_5stage(dt_c, ecg_c, aux_hr, spo2, hr_roll)
        comp_stale_rec.append(float(np.mean(c_c[s_idx:e_idx] == 4) * 100.0))
        tpc = int(np.sum((c_c != 0) & gt_c))
        fpc = int(np.sum((c_c != 0) & (~gt_c)))
        fnc = int(np.sum((c_c == 0) & gt_c))
        ov_rec = tpc / (tpc + fnc) * 100.0
        prec = tpc / (tpc + fpc) * 100.0 if (tpc + fpc) > 0 else 0.0
        comp_overall_rec.append(ov_rec)
        comp_f1.append(2 * prec * ov_rec / (prec + ov_rec) if (prec + ov_rec) > 0 else 0.0)

    out = {
        "n_trials": 20,
        "f5_random_window": {
            "stale_code_4_recall_mean": float(np.mean(stale_only_rec)),
            "stale_code_4_recall_sd": float(np.std(stale_only_rec, ddof=1)),
            "overall_fault_recall_mean": float(np.mean(stale_only_overall_rec)),
            "overall_fault_recall_sd": float(np.std(stale_only_overall_rec, ddof=1)),
        },
        "burst_plus_stale_random_window": {
            "stale_code_4_recall_mean": float(np.mean(comp_stale_rec)),
            "stale_code_4_recall_sd": float(np.std(comp_stale_rec, ddof=1)),
            "overall_fault_recall_mean": float(np.mean(comp_overall_rec)),
            "overall_fault_recall_sd": float(np.std(comp_overall_rec, ddof=1)),
            "overall_f1_mean": float(np.mean(comp_f1)),
            "overall_f1_sd": float(np.std(comp_f1, ddof=1)),
        },
    }
    (OUT_DIR / "randomized_stale_and_compound.json").write_text(json.dumps(out, indent=2))
    return out


def run_boundary_diagnostics():
    hr_const = np.full(6000, 80.0, dtype=np.float64)
    ecg_const, _ = hr_trajectory_to_beats(hr_const, fs=100.0)
    y3 = biquad_bandpass_3stage(ecg_const, fs=100.0)
    peaks = detect_rpeaks_adaptive(y3, r_min=30)
    hr_roll = rolling_9beat_hr(peaks, 6000, fs=100.0)
    dt_ms = np.full(6000, 10, dtype=np.int32)
    s0, s1 = 2000, 3500

    res_table = {}
    for off in [10.0, 11.0, 12.0, 13.0, 14.0]:
        fracs, means, meds = [], [], []
        for tr in range(20):
            aux_hr, spo2 = make_1hz_aux(hr_const, seed=800 + tr)
            aux_mod = aux_hr.copy()
            aux_mod[s0:s1] += off
            codes = classify_faults_5stage(dt_ms, ecg_const, aux_mod, spo2, hr_roll)
            fracs.append(float(np.mean(codes[s0:s1] == 5) * 100.0))
            diff = aux_mod[s0:s1] - hr_roll[s0:s1]
            means.append(float(np.mean(diff)))
            meds.append(float(np.median(diff)))
        res_table[f"+{off:.1f}"] = {
            "flag_pct_mean": float(np.mean(fracs)),
            "flag_pct_sd": float(np.std(fracs, ddof=1)),
            "diff_mean_bpm": float(np.mean(means)),
            "diff_median_bpm": float(np.median(meds)),
        }
    (OUT_DIR / "boundary_diagnostics.json").write_text(json.dumps(res_table, indent=2))
    return res_table


def main():
    elf_info = build_clean_elf()
    noise_rep = run_unified_noise_and_zero_lag_check()
    stale_rep = run_randomized_stale_and_compound()
    bound_rep = run_boundary_diagnostics()
    print("ELF:", elf_info["elf"], elf_info["sha256"][:16])
    print("Zero-lag sigma=0.08 FPR:", noise_rep["zero_lag_sigma_0.08"])
    print("Randomized stale & compound:", stale_rep)
    print("Boundary:", bound_rep)


if __name__ == "__main__":
    main()
