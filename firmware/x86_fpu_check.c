
#include <stdio.h>
#include <stdint.h>
#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"
int main(void) {
    BiquadFilterState f; RPeakDetectorState r;
    ECG_Filter_Init(&f); RPeak_Init(&r);
    uint32_t c = 0;
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {
        float y = ECG_Filter_Process(&f, ecg_data[i]);
        if (RPeak_Process(&r, y, i)) c++;
    }
    printf("X86_PEAKS=%u\n", c);
    return 0;
}
