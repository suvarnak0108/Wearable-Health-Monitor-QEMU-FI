#include <stdio.h>
#include <stdint.h>

#include "ecg_filter.h"
#include "r_peak.h"
#include "hr_estimator.h"

#define MAX_PEAKS 200

int main(void)
{
    FILE *file = fopen(
        "datasets/variable_noisy_ecg.csv", "r"
    );

    if (file == NULL)
    {
        printf("Dataset opening failed.\n");
        return 1;
    }

    ECGFilter filter;
    RPeakDetector detector;

    ECG_Filter_Init(&filter);
    RPeak_Init(&detector, 0.15f, 55);

    uint32_t peaks[MAX_PEAKS];
    uint32_t peak_count = 0;
    uint32_t peak_index;

    char line[256];
    float time, ecg, true_hr;

    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file))
    {
        if (sscanf(line, "%f,%f,%f",
                   &time, &ecg, &true_hr) != 3)
            continue;

        float filtered = ECG_Filter_Process(
            &filter, ecg
        );

        if (RPeak_Process(
            &detector, filtered, &peak_index
        ))
        {
            if (peak_count < MAX_PEAKS)
                peaks[peak_count++] = peak_index;
        }
    }

    fclose(file);

    printf("Total detected peaks: %u\n", peak_count);

    printf("\nHR ESTIMATION RESULTS\n");

    for (uint32_t start = 0; start < 60; start += 10)
    {
        float hr = HR_Estimate(
            peaks,
            peak_count,
            start * 100,
            (start + 10) * 100,
            100.0f
        );

        printf(
            "%u-%u seconds: %.2f bpm\n",
            start, start + 10, hr
        );
    }

    return 0;
}
