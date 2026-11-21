#ifndef R_PEAK_H
#define R_PEAK_H

#include <stdint.h>

typedef struct {
    float prev2;
    float prev1;
    float spki;
    float npki;
    float threshold;
    uint32_t sample_count;
    uint32_t last_peak;
    uint32_t refractory_samples;
    uint8_t initialized;
} RPeakDetector;

void RPeak_Init(
    RPeakDetector *detector,
    float initial_threshold,
    uint32_t refractory_samples
);

int RPeak_Process(
    RPeakDetector *detector,
    float sample,
    uint32_t *peak_index
);

#endif
