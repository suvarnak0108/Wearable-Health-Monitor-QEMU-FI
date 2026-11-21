#ifndef HR_ESTIMATOR_H
#define HR_ESTIMATOR_H

#include <stdint.h>

float HR_Estimate(
    const uint32_t *peaks,
    uint32_t count,
    uint32_t start_sample,
    uint32_t end_sample,
    float sampling_frequency
);

float HR_Estimate_FaultTolerant(
    const uint32_t *peaks,
    uint32_t count,
    uint32_t start_sample,
    uint32_t end_sample,
    float sampling_frequency,
    float last_valid_hr,
    uint32_t window_fault_count,
    uint32_t window_samples,
    float *sqi_out
);

#endif
