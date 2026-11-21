#include "fault_detector.h"

FaultType Fault_Detect(float sensor_hr, float ecg_hr)
{
    if (sensor_hr != sensor_hr || ecg_hr != ecg_hr)
        return FAULT_HR_RANGE;

    if (sensor_hr < 30.0f || sensor_hr > 200.0f)
        return FAULT_HR_RANGE;

    if (ecg_hr < 0.0f)
        return FAULT_HR_RANGE;

    float difference = sensor_hr - ecg_hr;

    if (difference > 10.0f || difference < -10.0f)
        return FAULT_HR_INCONSISTENCY;

    return FAULT_NONE;
}
