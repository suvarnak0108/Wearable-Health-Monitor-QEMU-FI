
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "ecg_filter.h"
#include "r_peak.h"

int main(void)
{
    FILE *file = fopen("datasets/high_noise_trial.csv", "r");

    if (file == NULL) {
        printf("Error opening dataset.\n");
        return 1;
    }

    ECGFilter filter;
    RPeakDetector detector;

    ECG_Filter_Init(&filter);
    RPeak_Init(&detector, 0.15f, 55);

    char line[256];

    float time, ecg, true_hr, filtered;
    float min_filtered = 1000000.0f;
    float max_filtered = -1000000.0f;
    double sum_squared = 0.0;

    uint32_t peak_index;
    uint32_t total_peaks = 0;
    uint32_t sample_total = 0;

    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {

        if (sscanf(line, "%f,%f,%f",
                   &time, &ecg, &true_hr) != 3) {
            continue;
        }

        filtered = ECG_Filter_Process(&filter, ecg);

        if (filtered < min_filtered)
            min_filtered = filtered;

        if (filtered > max_filtered)
            max_filtered = filtered;

        sum_squared += filtered * filtered;
        sample_total++;

        if (RPeak_Process(&detector, filtered, &peak_index)) {
            total_peaks++;
        }
    }

    fclose(file);

    printf("\nFiltered ECG statistics:\n");
    printf("Minimum: %.4f mV\n", min_filtered);
    printf("Maximum: %.4f mV\n", max_filtered);
    printf("Peak-to-peak: %.4f mV\n",
           max_filtered - min_filtered);

    if (sample_total > 0) {
        printf("RMS: %.4f mV\n",
               (float)sqrt(sum_squared / sample_total));
    }

    printf("\nTotal detected R-peaks: %u\n", total_peaks);

    return 0;
}
