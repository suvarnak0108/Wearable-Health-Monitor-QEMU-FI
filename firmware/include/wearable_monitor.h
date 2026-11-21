#ifndef WEARABLE_MONITOR_H
#define WEARABLE_MONITOR_H

#include <stdint.h>

typedef enum {
    FAULT_NONE = 0,
    FAULT_MISSING_SAMPLE = 1,
    FAULT_COMM_BURST = 2,
    FAULT_OUT_OF_RANGE = 3,
    FAULT_STALE_DATA = 4,
    FAULT_HR_INCONSISTENCY = 5
} FaultCode;

typedef struct {
    float z1[3];
    float z2[3];
} BiquadFilterState;

typedef struct {
    float spki;
    float npki;
    float threshold;
    uint32_t last_peak_idx;
    float prev_e1;
    float prev_e2;
} RPeakDetectorState;

typedef struct {
    float prev_ecg;
    float prev_hr;
    float prev_spo2;
    uint8_t stale_ecg;
    uint8_t stale_hr;
    uint8_t stale_spo2;
} FaultDetectorState;

void ECG_Filter_Init(BiquadFilterState *st);
float ECG_Filter_Process(BiquadFilterState *st, float x);

void RPeak_Init(RPeakDetectorState *st);
uint8_t RPeak_Process(RPeakDetectorState *st, float y_filt, uint32_t idx);

float HR_Estimate_Raw(const uint32_t *peaks, uint32_t count, float fs);
float HR_Estimate_RR_Filtered(const uint32_t *peaks, uint32_t count, float fs, float fallback_hr);
float HR_Estimate_Mitigated(const uint32_t *peaks, uint32_t count, float fs,
                            uint32_t fault_samples, uint32_t win_samples,
                            float last_valid_hr, float *out_sqi);

void Fault_Init(FaultDetectorState *st);
FaultCode Fault_Detect_Full(FaultDetectorState *st, uint32_t dt_ms, float ecg,
                            float sensor_hr, float spo2, float ecg_hr);

#endif
