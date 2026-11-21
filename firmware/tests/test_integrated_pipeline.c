#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "ecg_filter.h"
#include "r_peak.h"
#include "hr_estimator.h"
#include "fault_detector.h"

#define MAX_SAMPLES 6000
#define MAX_PEAKS 200
#define FS 100.0f
#define WINDOW_SIZE 1000

int main(void)
{
    FILE *data = fopen("datasets/variable_hr_fault_data.csv", "r");
    FILE *truth = fopen("datasets/variable_hr_ground_truth.csv", "r");

    if (!data || !truth) {
        printf("File opening failed.\n");
        return 1;
    }

    int faulty[MAX_SAMPLES] = {0};
    char line[256];
    unsigned int index;

    fgets(line, sizeof(line), truth);

    while (fscanf(truth, "%u", &index) == 1) {
        if (index < MAX_SAMPLES)
            faulty[index] = 1;
    }

    fclose(truth);

    ECGFilter filter;
    RPeakDetector detector;

    ECG_Filter_Init(&filter);
    RPeak_Init(&detector, 0.15f, 55);

    uint32_t peaks[MAX_PEAKS];
    uint32_t peak_count = 0;
    uint32_t peak_index;

    float sensor_hr[MAX_SAMPLES];

    fgets(line, sizeof(line), data);

    int sample = 0;
    float time, signal, true_hr, hr;

    while (fgets(line, sizeof(line), data) && sample < MAX_SAMPLES) {

        if (sscanf(line, "%f,%f,%f,%f",
                   &time, &signal, &true_hr, &hr) != 4)
            continue;

        sensor_hr[sample] = hr;

        float filtered = ECG_Filter_Process(&filter, signal);

        if (RPeak_Process(&detector, filtered, &peak_index)) {
            if (peak_count < MAX_PEAKS)
                peaks[peak_count++] = peak_index;
        }

        sample++;
    }

    fclose(data);

    int TP = 0, FP = 0, FN = 0, TN = 0;

    for (int start = 0; start < sample; start += WINDOW_SIZE) {

        int end = start + WINDOW_SIZE;
        if (end > sample)
            end = sample;

        float estimated_hr = HR_Estimate(
            peaks, peak_count, start, end, FS
        );

        for (int i = start; i < end; i++) {

            int detected = 0;

            if (estimated_hr > 0 &&
                Fault_Detect(sensor_hr[i], estimated_hr) != FAULT_NONE)
                detected = 1;

            if (detected && faulty[i]) TP++;
            else if (detected && !faulty[i]) FP++;
            else if (!detected && faulty[i]) FN++;
            else TN++;
        }

        printf("%d-%d seconds: ECG HR = %.2f bpm\n",
               start / 100, end / 100, estimated_hr);
    }

    float precision = (TP + FP) ? (float)TP / (TP + FP) : 0;
    float recall = (TP + FN) ? (float)TP / (TP + FN) : 0;
    float f1 = (precision + recall) ?
               2 * precision * recall / (precision + recall) : 0;

    printf("\nINTEGRATED PIPELINE RESULTS\n");
    printf("Samples processed: %d\n", sample);
    printf("Detected R-peaks: %u\n", peak_count);
    printf("TP: %d\nFP: %d\nFN: %d\nTN: %d\n",
           TP, FP, FN, TN);

    printf("Precision: %.2f%%\n", precision * 100);
    printf("Recall: %.2f%%\n", recall * 100);
    printf("F1-score: %.2f%%\n", f1 * 100);

    return 0;
}
