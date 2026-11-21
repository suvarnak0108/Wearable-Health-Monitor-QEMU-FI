import pandas as pd
import numpy as np

data = pd.read_csv("datasets/variable_hr_fault_data.csv")
truth = pd.read_csv("datasets/variable_hr_ground_truth.csv")

assert len(data) == 6000
assert len(truth) == 150

fault_flags = np.zeros(len(data), dtype=np.uint8)
fault_flags[truth["sample_index"].to_numpy(dtype=int)] = 1

with open("firmware/include/qemu_dataset.h", "w") as f:

    f.write("#ifndef QEMU_DATASET_H\n")
    f.write("#define QEMU_DATASET_H\n\n")

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
        data["ecg_mV"].to_numpy(),
        lambda x: f"{x:.8f}f"
    )

    write_array(
        "sensor_hr", "float",
        data["heart_rate_bpm"].to_numpy(),
        lambda x: f"{x:.2f}f"
    )

    write_array(
        "fault_ground_truth", "uint8_t",
        fault_flags,
        lambda x: str(int(x))
    )

    f.write("#endif\n")

print("QEMU dataset header generated.")
print("Samples:", len(data))
print("Injected faults:", int(fault_flags.sum()))
