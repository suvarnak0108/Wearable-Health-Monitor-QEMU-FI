#include <stdint.h>
#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"
static void sh_w0(const char *s) {
    __asm__ volatile ("mov r0, #0x04\nmov r1, %0\nbkpt #0xAB\n" : : "r"(s) : "r0", "r1", "memory");
}
static void sh_exit(void) {
    __asm__ volatile ("mov r0, #0x18\nldr r1, =0x20026\nbkpt #0xAB\n" : : : "r0", "r1", "memory");
    while (1) {}
}
int main(void) {
    BiquadFilterState f;
    ECG_Filter_Init(&f);
    const char *hex = "0123456789abcdef";
    char buf[10]; buf[8] = 10; buf[9] = 0;
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {
        float y = ECG_Filter_Process(&f, ecg_data[i]);
        uint32_t bits; __builtin_memcpy(&bits, &y, 4);
        for (int b = 7; b >= 0; b--) { buf[b] = hex[bits & 0xFU]; bits >>= 4; }
        sh_w0(buf);
    }
    sh_exit();
    return 0;
}
