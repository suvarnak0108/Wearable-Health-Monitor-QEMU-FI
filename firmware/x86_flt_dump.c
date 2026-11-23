
#include <stdio.h>
#include <stdint.h>
#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"
int main(void) {
    BiquadFilterState f;
    ECG_Filter_Init(&f);
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {
        float y = ECG_Filter_Process(&f, ecg_data[i]);
        uint32_t bits; __builtin_memcpy(&bits, &y, 4);
        printf("%08x\n", bits);
    }
    return 0;
}
