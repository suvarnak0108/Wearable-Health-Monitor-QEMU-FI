import pandas as pd
import numpy as np
import subprocess
import re
from pathlib import Path

ROOT = Path.cwd()

DATA_FILE = ROOT / "datasets/independent_validation_data.csv"
HEADER = ROOT / "firmware/include/qemu_dataset_independent.h"
ELF = ROOT / "firmware/independent_trial.elf"

df = pd.read_csv(DATA_FILE)

results = []

groups = list(df.groupby(["noise_std", "trial"], sort=True))

print("Starting independent QEMU validation")
print("Total trials:", len(groups))

for number, ((noise, trial_no), trial) in enumerate(groups, 1):

    trial = trial.sort_values("time_s").reset_index(drop=True)

    assert len(trial) == 6000
    assert int(trial["fault"].sum()) == 150

    ecg = trial["ecg_mV"].to_numpy()
    hr = trial["heart_rate_bpm"].to_numpy()
    labels = trial["fault"].to_numpy(dtype=np.uint8)

    with open(HEADER, "w") as f:

        f.write("#ifndef QEMU_DATASET_INDEPENDENT_H\n")
        f.write("#define QEMU_DATASET_INDEPENDENT_H\n")
        f.write("#include <stdint.h>\n")
        f.write("#define DATASET_SIZE 6000\n")
        f.write("#define SAMPLE_RATE 100.0f\n\n")

        def write_array(name, ctype, values, formatter):
            f.write(
                f"static const {ctype} "
                f"{name}[DATASET_SIZE] = {{\n"
            )

            for i in range(0, len(values), 10):
                row = ", ".join(
                    formatter(x) for x in values[i:i+10]
                )
                f.write("    " + row + ",\n")

            f.write("};\n\n")

        write_array(
            "ecg_data", "float", ecg,
            lambda x: f"{x:.8f}f"
        )

        write_array(
            "sensor_hr", "float", hr,
            lambda x: f"{x:.2f}f"
        )

        write_array(
            "fault_ground_truth", "uint8_t", labels,
            lambda x: str(int(x))
        )

        f.write("#endif\n")

    compile_command = [
        "arm-none-eabi-gcc",
        "-mcpu=cortex-m4",
        "-mthumb",
        "-mfpu=fpv4-sp-d16",
        "-mfloat-abi=hard",
        "-ffreestanding",
        "-nostdlib",
        "-Wall",
        "-Wextra",
        "-O1",
        "-I", "firmware/include",
        "-Wl,-T,firmware/linker/mps2_an386.ld",
        "firmware/startup/startup.s",
        "firmware/src/main.c",
        "firmware/src/ecg_filter.c",
        "firmware/src/r_peak.c",
        "firmware/src/hr_estimator.c",
        "firmware/src/fault_detector.c",
        "-lgcc",
        "-o", str(ELF)
    ]

    build = subprocess.run(
        compile_command,
        capture_output=True,
        text=True
    )

    if build.returncode != 0:
        print("Compilation failed:", noise, trial_no)
        print(build.stderr)
        raise SystemExit(1)

    qemu_command = [
        "qemu-system-arm",
        "-M", "mps2-an386",
        "-cpu", "cortex-m4",
        "-kernel", str(ELF),
        "-nographic",
        "-semihosting-config",
        "enable=on,target=native"
    ]

    run = subprocess.run(
        qemu_command,
        capture_output=True,
        text=True,
        timeout=15
    )

    output = run.stdout + run.stderr

    def get_value(pattern):
        match = re.search(pattern, output)
        if not match:
            raise RuntimeError(
                f"Could not parse {pattern}\n{output}"
            )
        return int(match.group(1))

    rpeaks = get_value(r"Detected R-peaks:\s*(\d+)")
    tp = get_value(r"TP:\s*(\d+)")
    fp = get_value(r"FP:\s*(\d+)")
    fn = get_value(r"FN:\s*(\d+)")
    tn = get_value(r"TN:\s*(\d+)")

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    fpr = fp / (fp + tn) if fp + tn else 0

    results.append({
        "noise_std": noise,
        "trial": trial_no,
        "r_peaks": rpeaks,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "FPR": fpr
    })

    print(
        f"[{number}/80] Noise={noise:.2f}, Trial={trial_no}: "
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}, "
        f"F1={f1*100:.2f}%"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    ROOT / "results/qemu_independent_80_trials.csv",
    index=False
)

summary = results_df.groupby("noise_std").agg({
    "precision": "mean",
    "recall": "mean",
    "F1": "mean",
    "FPR": "mean",
    "r_peaks": "mean"
}).reset_index()

summary.to_csv(
    ROOT / "results/qemu_independent_summary.csv",
    index=False
)

print("\nALL 80 QEMU TRIALS COMPLETED")
print(summary.to_string(index=False))
