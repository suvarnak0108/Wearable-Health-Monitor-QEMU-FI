from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.signal import butter, sosfilt

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_fig1(out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((2, 3), 42, 46, boxstyle="round,pad=0.8",
                                  facecolor="#eef4fb", edgecolor="#4a79b8", linewidth=1.5))
    ax.text(23, 46.5, "HOST ENVIRONMENT (Python 3 / WSL Ubuntu)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1e3f66")

    left_boxes = [
        (33.5, "1. Multi-Source Physiological Dataset Generator",
         "55.8–134.5 BPM ECG + 1 Hz Aux + 10 MIT-BIH Records"),
        (19.5, "2. Sample & Front-End Fault Injector (F1–F9)",
         "Noise, Dropouts, Bursts, Range, Stale, Drift (1–20%) + Ramp/Hum"),
        (5.5, "6. Statistical & Metrics Evaluator",
         "Safety Properties (P1–P4), AAMI EC57, MAE & SQI"),
    ]
    for y, title, sub in left_boxes:
        ax.add_patch(FancyBboxPatch((5, y), 36, 10.5, boxstyle="round,pad=0.6",
                                      facecolor="white", edgecolor="#6d95c9", linewidth=1.2))
        ax.text(23, y + 6.8, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#111111")
        ax.text(23, y + 3.0, sub, ha="center", va="center", fontsize=7.6, color="#333333")

    ax.add_patch(FancyBboxPatch((56, 3), 42, 46, boxstyle="round,pad=0.8",
                                  facecolor="#edf7f1", edgecolor="#3f8f62", linewidth=1.5))
    ax.text(77, 46.5, "TARGET EMULATION (QEMU ARM Cortex-M4)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1e5938")

    right_boxes = [
        (33.5, "3. Cascaded Biquad & Adaptive R-Peak DSP",
         "3-Stage 5–15 Hz Butterworth + Quadratic 300 ms R-Peak"),
        (19.5, "4. 5-Stage Classifier & RR/SQI Recovery",
         "Fault Isolation + Physiological RR Filter + SQI Gate"),
        (5.5, "5. ARM Semihosting & Stack Telemetry",
         "Bare-Metal mps2-an386 (2,312 B Code / 3,080–3,096 B HWM)"),
    ]
    for y, title, sub in right_boxes:
        ax.add_patch(FancyBboxPatch((59, y), 36, 10.5, boxstyle="round,pad=0.6",
                                      facecolor="white", edgecolor="#65ad84", linewidth=1.2))
        ax.text(77, y + 6.8, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#111111")
        ax.text(77, y + 3.0, sub, ha="center", va="center", fontsize=7.6, color="#333333")

    for y_top, y_bot in [(33.5, 30.6), (19.5, 16.6)]:
        ax.annotate("", xy=(23, y_bot), xytext=(23, y_top),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
        ax.annotate("", xy=(77, y_bot), xytext=(77, y_top),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))

    ax.annotate("", xy=(58.6, 38.7), xytext=(41.4, 24.7),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#222222"))
    ax.text(50, 34.0, "Static C Header\nTest Vectors\n(.h Arrays)", ha="center", va="center",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888"))

    ax.annotate("", xy=(41.4, 10.7), xytext=(58.6, 10.7),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#222222"))
    ax.text(50, 14.5, "Semihosted\nMetrics & Logs", ha="center", va="center",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888888"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_fig2(out_path: Path):
    fs = 100.0
    t = np.arange(1000) / fs
    rng = np.random.default_rng(42)

    beats = np.arange(0.4, 10.0, 60.0 / 75.0)
    amp = [0.12, -0.15, 1.00, -0.22, 0.28]
    off = [-0.16, -0.035, 0.0, 0.035, 0.22]
    wid = [0.018, 0.007, 0.0085, 0.007, 0.032]
    ecg_clean = np.zeros_like(t)
    for tk in beats:
        for a, d, w in zip(amp, off, wid):
            ecg_clean += a * np.exp(-0.5 * ((t - tk - d) / w) ** 2)

    sos = butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    ecg_low_in = ecg_clean + rng.normal(0.0, 0.08, size=len(t))
    ecg_high_in = ecg_clean + rng.normal(0.0, 0.35, size=len(t))
    filt_low = sosfilt(sos, ecg_low_in)
    filt_high = sosfilt(sos, ecg_high_in)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.2), dpi=300, sharex=True)
    ax1.plot(t, ecg_low_in, lw=0.8, alpha=0.85, color="#3b78b8", label="Low-Noise ECG (0.08 mV)")
    ax1.plot(t, filt_low, lw=1.1, color="#1f7a3d", label="Bandpass Filtered (5–15 Hz)")
    ax1.set_title("(a) Low-Noise ECG Regime (0.08 mV) — 0.000 BPM RR-Filtered HR MAE", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Amplitude (mV)", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=7.5)

    ax2.plot(t, ecg_high_in, lw=0.75, alpha=0.65, color="#8c8c8c", label="High-Noise ECG (0.35 mV)")
    ax2.plot(t, filt_high, lw=1.1, color="#b22222", label="Bandpass Filtered (5–15 Hz)")
    ax2.set_title("(b) Severe Broadband Noise Regime (0.35 mV) — 300 ms Refractory Noise Admission Limit", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Time (seconds)", fontsize=8.5)
    ax2.set_ylabel("Amplitude (mV)", fontsize=8.5)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_fig3(out_path: Path):
    labels = ["0.08", "0.15", "0.25", "0.35"]
    peaks = [79.00, 79.30, 89.75, 109.85]
    maes = [0.000, 0.012, 7.881, 41.257]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=300)

    bars1 = ax1.bar(labels, peaks, color="#2b5c8f", width=0.55, label="Detected R-Peaks (Mean)")
    ax1.axhline(79.0, color="#c87d32", linestyle="--", linewidth=1.2, label="True Beat Count (79)")
    ax1.set_ylim(0, 130)
    ax1.set_title("(a) Detected Peaks vs. True Beats (300 ms Refractory)", fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Gaussian Noise Std. Dev. (mV)", fontsize=8.5)
    ax1.set_ylabel("Mean Detected R-Peaks", fontsize=8.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=7.5)
    for b, val in zip(bars1, peaks):
        ax1.text(b.get_x() + b.get_width() / 2, val + 2.2, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    bars2 = ax2.bar(labels, maes, color="#238b54", width=0.55, label="RR-Filtered HR MAE (BPM)")
    ax2.set_ylim(0, 48)
    ax2.set_title("(b) Windowed HR Error Under White Noise", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Gaussian Noise Std. Dev. (mV)", fontsize=8.5)
    ax2.set_ylabel("RR-Filtered HR MAE (BPM)", fontsize=8.5)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=7.5)
    for b, val in zip(bars2, maes):
        ax2.text(b.get_x() + b.get_width() / 2, val + 0.9, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    make_fig1(OUT_DIR / "fig1_architecture.png")
    make_fig2(OUT_DIR / "fig2_waveform_comparison.png")
    make_fig3(OUT_DIR / "fig3_fixed_vs_adaptive.png")
    print("Regenerated fig1, fig2, and fig3.")


if __name__ == "__main__":
    main()
