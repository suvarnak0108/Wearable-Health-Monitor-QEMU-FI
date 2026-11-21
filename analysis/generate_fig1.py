import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=300)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 5.4)
ax.axis("off")

# Outer Containers (Host vs Target)
host_box = FancyBboxPatch(
    (0.2, 0.25), 4.65, 4.9,
    boxstyle="round,pad=0.06,rounding_size=0.16",
    facecolor="#f2f6fa", edgecolor="#234d76", linewidth=2.0
)
target_box = FancyBboxPatch(
    (6.45, 0.25), 4.85, 4.9,
    boxstyle="round,pad=0.06,rounding_size=0.16",
    facecolor="#f1f8f3", edgecolor="#26613a", linewidth=2.0
)
ax.add_patch(host_box)
ax.add_patch(target_box)

# Container Headers (sized cleanly inside boxes)
ax.text(2.525, 4.82, "HOST ENVIRONMENT (Python 3 / WSL Ubuntu)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1b3a57")
ax.text(8.875, 4.82, "TARGET EMULATION (QEMU ARM Cortex-M4)",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1d4d2e")

def draw_step(x, y, w, h, title, subtitle, bg="#ffffff", edge="#234d76"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=bg, edgecolor=edge, linewidth=1.4
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.63, title,
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#111111")
    ax.text(x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=8.3, color="#333333")

# Left Column: Host Steps
draw_step(0.45, 3.30, 4.15, 1.12,
          "1. Multi-Source Physiological Dataset Generator",
          "Synthetic 100 Hz ECG/HR/SpO2 + 10 MIT-BIH Patient Records",
          bg="#ffffff", edge="#234d76")

draw_step(0.45, 1.88, 4.15, 1.12,
          "2. 8-Class & Compound Fault Injector",
          "Noise, Dropouts, Bursts, Range, Stale, Drift & Sweep (1-20%)",
          bg="#ffffff", edge="#234d76")

draw_step(0.45, 0.46, 4.15, 1.12,
          "6. Verification Oracle & Statistical Evaluator",
          "Safety Properties (P1-P4), Confusion Matrix, HR MAE & SQI",
          bg="#e5eef7", edge="#234d76")

# Right Column: Target Steps (Bare-Metal Embedded C on mps2-an386)
draw_step(6.70, 3.30, 4.35, 1.12,
          "3. Cascaded Biquad & Adaptive R-Peak DSP",
          "3-Stage 5-15 Hz Butterworth + Quadratic Adaptive Threshold",
          bg="#ffffff", edge="#26613a")

draw_step(6.70, 1.88, 4.35, 1.12,
          "4. 5-Stage Classifier & Active HR Recovery",
          "Fault Isolation + Physiological RR Filter + SQI Confidence",
          bg="#ffffff", edge="#26613a")

draw_step(6.70, 0.46, 4.35, 1.12,
          "5. ARM Semihosting Telemetry (BKPT 0xAB)",
          "Bare-Metal mps2-an386 Execution Log (2,312 B Core Code)",
          bg="#e3f1e8", edge="#26613a")

# Vertical Arrows inside Host & Target
arrow_kw = dict(arrowstyle="-|>,head_width=0.25,head_length=0.35", lw=1.8, color="#333333")
ax.annotate("", xy=(2.525, 3.03), xytext=(2.525, 3.28), arrowprops=arrow_kw)
ax.annotate("", xy=(2.525, 1.61), xytext=(2.525, 1.86), arrowprops=arrow_kw)
ax.annotate("", xy=(8.875, 3.03), xytext=(8.875, 3.28), arrowprops=arrow_kw)
ax.annotate("", xy=(8.875, 1.61), xytext=(8.875, 1.86), arrowprops=arrow_kw)

# Cross-Boundary Arrows with Clean Pill Badges (No Overlap)
cross_kw = dict(arrowstyle="-|>,head_width=0.28,head_length=0.38", lw=2.0, color="#222222")

# Forward path: Step 2 (y=2.44) -> Step 3 (y=3.86)
ax.annotate("", xy=(6.66, 3.86), xytext=(4.64, 2.44), arrowprops=cross_kw)
ax.text(5.65, 3.15, "Static C Header\nTest Vectors\n(.h Arrays)",
        ha="center", va="center", fontsize=8.3, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#fffdf5", edgecolor="#888888", lw=1.0))

# Return path: Step 5 (y=1.02) -> Step 6 (y=1.02)
ax.annotate("", xy=(4.64, 1.02), xytext=(6.66, 1.02), arrowprops=cross_kw)
ax.text(5.65, 1.02, "Semihosted\nMetrics & Logs",
        ha="center", va="center", fontsize=8.3, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#fffdf5", edgecolor="#888888", lw=1.0))

plt.tight_layout(pad=0.2)
plt.savefig("results/fig1_architecture.png", dpi=300)
plt.close()
print("Saved clean results/fig1_architecture.png!")
