#ifndef FAULT_DETECTOR_H
#define FAULT_DETECTOR_H

#include <stdint.h>

typedef enum {
    FAULT_NONE = 0,
    FAULT_HR_RANGE,
    FAULT_HR_INCONSISTENCY
} FaultType;

FaultType Fault_Detect(float sensor_hr, float ecg_hr);

#endif
