import pandas as pd
import numpy as np

NOISE_LEVEL = 0.08
TRIAL_NUMBER = 1

df = pd.read_csv("datasets/independent_validation_data.csv")

trial = df[
    (df["noise_std"] == NOISE_LEVEL) &
    (df["trial"] == TRIAL_NUMBER)
].copy()

trial = trial.sort_values("time_s").reset_index(drop=True)

assert len(trial) == 6000
assert trial["fault"].sum() == 150

with open("firmware/include/qemu_dataset_independent.h", "w") as f:

    f.write("#ifndef QEMU_DATASET_INDEPENDENT_H\n")
    f.write("#define QEMU_DATASET_INDEPENDENT_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write("#define DATASET_SIZE 6000\n")
    f.write("#define SAMPLE_RATE 100.0f\n\n")

    def write_array(name, ctype, values, formatter):
        f.write(f"static const {ctype} {name}[DATASET_SIZE] = {{\n")

        for i in range(0, len(values), 10):
            row = ", ".join(formatter(x) for x in values[i:i+10])
            f.write("    " + row + ",\n")

        f.write("};\n\n")

    write_array(
        "ecg_data", "float",
        trial["ecg_mV"].to_numpy(),
        lambda x: f"{x:.8f}f"
    )

    write_array(
        "sensor_hr", "float",
        trial["heart_rate_bpm"].to_numpy(),
        lambda x: f"{x:.2f}f"
    )

    write_array(
        "fault_ground_truth", "uint8_t",
        trial["fault"].to_numpy(dtype=np.uint8),
        lambda x: str(int(x))
    )

    f.write("#endif\n")

print("Independent QEMU dataset generated.")
print("Noise level:", NOISE_LEVEL)
print("Trial:", TRIAL_NUMBER)
print("Samples:", len(trial))
print("Injected faults:", int(trial["fault"].sum()))
print("Normal samples:", int((trial["fault"] == 0).sum()))
