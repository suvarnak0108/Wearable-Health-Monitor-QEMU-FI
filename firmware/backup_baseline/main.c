#include <stdint.h>

#include "ecg_filter.h"
#include "r_peak.h"
#include "hr_estimator.h"
#include "fault_detector.h"
#include "qemu_dataset_independent.h"

#define MAX_PEAKS 200
#define WINDOW_SIZE 1000

static void semihost_write0(const char *message)
{
    register int operation __asm__("r0") = 0x04;
    register const char *argument __asm__("r1") = message;

    __asm__ volatile (
        "bkpt 0xAB"
        : "+r"(operation)
        : "r"(argument)
        : "memory"
    );
}
static void semihost_exit(void)
{
    register int operation __asm__("r0") = 0x18;
    register uint32_t reason __asm__("r1") = 0x20026;

    __asm__ volatile (
        "bkpt 0xAB"
        : "+r"(operation)
        : "r"(reason)
        : "memory"
    );
}
static void print_uint(uint32_t value)
{
    char buffer[12];
    int i = 0;

    do {
        buffer[i++] = '0' + (value % 10);
        value /= 10;
    } while (value);

    buffer[i] = '\0';

    for (int j = 0; j < i / 2; j++) {
        char temp = buffer[j];
        buffer[j] = buffer[i - j - 1];
        buffer[i - j - 1] = temp;
    }

    semihost_write0(buffer);
}

static void print_percent(uint32_t numerator, uint32_t denominator)
{
    if (denominator == 0) {
        semihost_write0("N/A");
        return;
    }

    uint32_t value = (numerator * 10000U) / denominator;

    print_uint(value / 100);
    semihost_write0(".");

    char fraction[3] = {
        '0' + ((value % 100) / 10),
        '0' + (value % 10),
        '\0'
    };

    semihost_write0(fraction);
    semihost_write0("%");
}

int main(void)
{
    ECGFilter filter;
    RPeakDetector detector;

    uint32_t peaks[MAX_PEAKS];
    uint32_t peak_count = 0;
    uint32_t peak_index;

    uint32_t TP = 0;
    uint32_t FP = 0;
    uint32_t FN = 0;
    uint32_t TN = 0;

    ECG_Filter_Init(&filter);
    RPeak_Init(&detector, 0.15f, 55);

    semihost_write0("\nQEMU WEARABLE HEALTH MONITOR\n");
    semihost_write0("Processing 6000 ECG samples...\n");

    /* ECG filtering and R-peak detection */
    for (uint32_t i = 0; i < DATASET_SIZE; i++) {

        float filtered = ECG_Filter_Process(&filter, ecg_data[i]);

        if (RPeak_Process(&detector, filtered, &peak_index)) {

            if (peak_count < MAX_PEAKS)
                peaks[peak_count++] = peak_index;
        }
    }

    /* HR estimation and fault classification */
    for (uint32_t start = 0; start < DATASET_SIZE;
         start += WINDOW_SIZE) {

        uint32_t end = start + WINDOW_SIZE;

        if (end > DATASET_SIZE)
            end = DATASET_SIZE;

        float estimated_hr = HR_Estimate(
            peaks,
            peak_count,
            start,
            end,
            SAMPLE_RATE
        );

        semihost_write0("Window ");
        print_uint(start / 100);
        semihost_write0("-");

        print_uint(end / 100);
        semihost_write0(" seconds: HR x100 = ");

        if (estimated_hr > 0.0f)
            print_uint((uint32_t)(estimated_hr * 100.0f));
        else
            semihost_write0("Unavailable");

        semihost_write0("\n");

        for (uint32_t i = start; i < end; i++) {

            int detected = 0;

            if (estimated_hr > 0.0f &&
                Fault_Detect(sensor_hr[i], estimated_hr) != FAULT_NONE)
                detected = 1;

            if (detected && fault_ground_truth[i])
                TP++;
            else if (detected && !fault_ground_truth[i])
                FP++;
            else if (!detected && fault_ground_truth[i])
                FN++;
            else
                TN++;
        }
    }

    semihost_write0("\nQEMU RESULTS\n");

    semihost_write0("Samples: ");
    print_uint(DATASET_SIZE);

    semihost_write0("\nDetected R-peaks: ");
    print_uint(peak_count);

    semihost_write0("\nTP: ");
    print_uint(TP);

    semihost_write0("\nFP: ");
    print_uint(FP);

    semihost_write0("\nFN: ");
    print_uint(FN);

    semihost_write0("\nTN: ");
    print_uint(TN);

    semihost_write0("\nPrecision: ");
    print_percent(TP, TP + FP);

    semihost_write0("\nRecall: ");
    print_percent(TP, TP + FN);

    semihost_write0("\nF1-score: ");

    if ((2 * TP + FP + FN) == 0) {
        semihost_write0("N/A");
    } else {
        print_percent(2 * TP, 2 * TP + FP + FN);
    }

    semihost_write0("\nQEMU firmware execution completed.\n");

semihost_exit();

while (1) {
    __asm__ volatile ("wfi");
}
}
