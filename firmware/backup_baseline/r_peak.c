
#include <stdio.h>
#include "r_peak.h"

void RPeak_Init(RPeakDetector *detector,
                float threshold,
                uint32_t refractory_samples)
{
    detector->prev2 = 0.0f;
    detector->prev1 = 0.0f;
    detector->sample_count = 0;
    detector->last_peak = 0;
    detector->refractory_samples = refractory_samples;
    detector->threshold = threshold;
    detector->initialized = 0;
}

int RPeak_Process(RPeakDetector *detector,
                  float sample,
                  uint32_t *peak_index)
{
    uint32_t current_index = detector->sample_count++;

    if (!detector->initialized) {
        detector->prev1 = sample;
        detector->initialized = 1;
        return 0;
    }

    if (sample < detector->prev1 &&
        detector->prev1 >= detector->prev2 &&
        detector->prev1 > detector->threshold) {

        uint32_t candidate_index = current_index - 1;

        if (candidate_index - detector->last_peak >=
            detector->refractory_samples) {

            printf("Peak amplitude: %.4f mV\n",
                   detector->prev1);

            detector->last_peak = candidate_index;
            *peak_index = candidate_index;

            detector->prev2 = detector->prev1;
            detector->prev1 = sample;

            return 1;
        }
    }

    detector->prev2 = detector->prev1;
    detector->prev1 = sample;

    return 0;
}

