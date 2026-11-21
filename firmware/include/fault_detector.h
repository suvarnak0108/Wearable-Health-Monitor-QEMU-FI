#ifndef FAULT_DETECTOR_H
#define FAULT_DETECTOR_H

#include <stdint.h>

typedef enum {
    FAULT_NONE = 0,
    FAULT_MISSING_SAMPLE,
    FAULT_COMM_BURST,
    FAULT_OUT_OF_RANGE,
    FAULT_STALE_DATA,
    FAULT_HR_INCONSISTENCY
} FaultType;

typedef struct {
    float prev_ecg;
    float prev_hr;
    float prev_spo2;
    uint32_t stale_count;
    uint8_t has_prev;
} FaultMonitorState;

void Fault_Init(FaultMonitorState *state);

FaultType Fault_Detect_Full(
    FaultMonitorState *state,
    uint32_t dt_ms,
    float ecg_mv,
    float sensor_hr,
    float spo2,
    float ecg_hr
);

#endif
