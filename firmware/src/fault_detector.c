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
    /* 1. Communication burst & single-sample dropout checks */
    if (dt_ms >= 50U) {
        st->stale_ecg = st->stale_hr = st->stale_spo2 = 0U;
        return FAULT_COMM_BURST;
    }
    if (dt_ms > 15U) {
        st->stale_ecg = st->stale_hr = st->stale_spo2 = 0U;
        return FAULT_MISSING_SAMPLE;
    }

    /* 2. Out-of-range and NaN guard */
    if (sensor_hr != sensor_hr || spo2 != spo2 || ecg != ecg ||
        sensor_hr < 30.0f || sensor_hr > 200.0f ||
        spo2 < 70.0f || spo2 > 100.0f) {
        return FAULT_OUT_OF_RANGE;
    }

    /* 3. Per-channel stale register detection (independent freeze counters) */
    st->stale_ecg  = (fabsf_local(ecg - st->prev_ecg) < 1e-6f) ? (st->stale_ecg + 1U) : 0U;
    st->stale_hr   = (fabsf_local(sensor_hr - st->prev_hr) < 1e-4f) ? (st->stale_hr + 1U) : 0U;
    st->stale_spo2 = (fabsf_local(spo2 - st->prev_spo2) < 1e-4f) ? (st->stale_spo2 + 1U) : 0U;

    st->prev_ecg = ecg;
    st->prev_hr = sensor_hr;
    st->prev_spo2 = spo2;

    /* Flag if ECG or HR freezes for >= 5 samples (50 ms) or SpO2 freezes >= 200 samples (2.0 s) */
    if (st->stale_ecg >= 5U || st->stale_hr >= 5U || st->stale_spo2 >= 200U) {
        return FAULT_STALE_DATA;
    }

    /* 4. Cross-modal HR consistency check */
    if (ecg_hr > 0.0f && fabsf_local(sensor_hr - ecg_hr) > 10.0f) {
        return FAULT_HR_INCONSISTENCY;
    }

    return FAULT_NONE;
}
