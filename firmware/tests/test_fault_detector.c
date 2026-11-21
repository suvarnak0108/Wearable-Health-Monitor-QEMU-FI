#include <stdio.h>
#include "fault_detector.h"

int main(void)
{
    float sensor_hr[] = {75, 250, 110, 85};
    float ecg_hr[]    = {75, 75, 75, 85};

    int n = 4;

    for (int i = 0; i < n; i++)
    {
        FaultType result = Fault_Detect(sensor_hr[i], ecg_hr[i]);

        printf("Sensor HR: %.1f, ECG HR: %.1f -> ",
               sensor_hr[i], ecg_hr[i]);

        if (result == FAULT_NONE)
            printf("Normal\n");
        else if (result == FAULT_HR_RANGE)
            printf("HR Range Fault\n");
        else
            printf("HR Inconsistency Fault\n");
    }

    return 0;
}
