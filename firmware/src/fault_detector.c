#include "wearable_monitor.h"

static inline float fabsf_local(float x) { return (x < 0.0f) ? -x : x; }

void Fault_Init(FaultDetectorState *st) {
    st->prev_ecg = 0.0f;
    st->prev_hr = 0.0f;
    st->prev_spo2 = 0.0f;
    st->stale_ecg = 0U;
    st->stale_hr = 0U;
    st->stale_spo2 = 0U;
}

FaultCode Fault_Detect_Full(FaultDetectorState *st, uint32_t dt_ms, float ecg,
                            float sensor_hr, float spo2, float ecg_hr) {
    if (dt_ms >= 50U) {
        st->stale_ecg = 0U;
        return FAULT_COMM_BURST;
    }
    if (dt_ms > 15U) {
        st->stale_ecg = 0U;
        return FAULT_MISSING_SAMPLE;
    }
    if (sensor_hr != sensor_hr || spo2 != spo2 || ecg != ecg ||
        sensor_hr < 30.0f || sensor_hr > 200.0f ||
        spo2 < 70.0f || spo2 > 100.0f) {
        return FAULT_OUT_OF_RANGE;
    }
    /* High-rate (100 Hz) ECG ADC freeze >= 150 samples (1.5 s > 1 full cardiac cycle) */
    if (fabsf_local(ecg - st->prev_ecg) < 1e-6f) {
        if (st->stale_ecg < 250U) st->stale_ecg++;
    } else {
        st->stale_ecg = 0U;
    }
    st->prev_ecg = ecg;
    if (st->stale_ecg >= 150U) {
        return FAULT_STALE_DATA;
    }
    /* Cross-modal HR consistency check */
    if (ecg_hr > 0.0f && fabsf_local(sensor_hr - ecg_hr) > 12.0f) {
        return FAULT_HR_INCONSISTENCY;
    }
    return FAULT_NONE;
}
