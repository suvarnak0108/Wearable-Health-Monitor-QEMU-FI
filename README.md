# Wearable Health Monitor QEMU Fault-Injection Testbed

Host-target evaluation pipeline for bare-metal Embedded C wearable cardiac monitoring firmware on an emulated ARM Cortex-M4 (`mps2-an386`) using Python and QEMU.

## Repository Layout

* `firmware/` — Bare-metal Embedded C DSP, adaptive R-peak detector, 5-stage fault classifier, and RR/SQI recovery
* `simulation/` — Unified Python runner (`run_unified_master_pipeline.py`), verification suite (`run_complete_verification_suite.py`), fault injectors, and QEMU harness
* `datasets/` — Synthetic profiles, PhysioNet MIT-BIH / NSTDB preprocessing & C header generators
* `analysis/` — Statistical evaluation, beat-matching (EC57), and plot/table generation scripts
* `results/` — Experimental logs, audit outputs (`results/audit_v3/`), tables, and figures

## Pinned Tool Versions

* **OS / Host:** Windows WSL Ubuntu 26.04
* **Python:** 3.12.3 (`numpy` 2.5.4, `scipy` 1.18.1, `wfdb` 4.3.1)
* **Toolchain:** `arm-none-eabi-gcc` 13.2.1 (`15:13.2.rel1-2`)
* **Emulator:** `qemu-system-arm` 8.2.2
* **Native compiler:** `gcc` 13.3.0

## Reproducing Results

1. Generate dataset headers and run the unified QEMU evaluation suite:
   ```bash
   python3 simulation/run_unified_master_pipeline.py
