# Wearable Health Monitor QEMU Fault-Injection Testbed

Host-target evaluation pipeline for bare-metal Embedded C wearable cardiac monitoring firmware on an emulated ARM Cortex-M4 (`mps2-an386`) using Python and QEMU.

## Repository Layout
- `firmware/` — Bare-metal Embedded C DSP, adaptive R-peak detector, 5-stage fault classifier, and RR/SQI recovery
- `simulation/` — Unified Python runner (`run_unified_master_pipeline.py`), fault injectors, and QEMU harness
- `datasets/` — Synthetic profiles, PhysioNet MIT-BIH / NSTDB preprocessing & C header generators
- `analysis/` — Statistical evaluation, beat-matching (EC57), and plot/table generation scripts
- `results/` — Experimental logs, tables, and figures

## Pinned Tool Versions
- OS / Host: Windows WSL Ubuntu `<exact version, e.g., 24.04 / 26.04>`
- Python: `<e.g., 3.12.x>` (`numpy==...`, `scipy==...`, `pandas==...`)
- Toolchain: `arm-none-eabi-gcc <exact version>` (`-O2 -ffreestanding -nostdlib -ffp-contract=off -fstack-usage -mfpu=fpv4-sp-d16 -mfloat-abi=hard`)
- Emulator: `qemu-system-arm <exact version, e.g., 8.2.x / 9.x>` (`mps2-an386`)
- Native compiler: `gcc <exact version>`

## Reproducing Results
1. Generate dataset headers and run the QEMU suite:
   `python3 simulation/run_unified_master_pipeline.py`
2. Run analysis and regenerate tables/figures:
   `python3 analysis/<script_name>.py`

## License
MIT License (see `LICENSE`).
