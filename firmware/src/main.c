#include "wearable_monitor.h"
#include "qemu_dataset_independent.h"

#define MAX_PEAKS 250U
#define WIN_SAMPLES 1000U

static void semihost_write0(const char *s) {
    __asm__ volatile (
        "mov r0, #0x04\n"
        "mov r1, %0\n"
        "bkpt #0xAB\n"
        : : "r"(s) : "r0", "r1", "memory"
    );
}

static void semihost_exit(void) {
    __asm__ volatile (
        "mov r0, #0x18\n"
        "ldr r1, =0x20026\n"
        "bkpt #0xAB\n"
        : : : "r0", "r1", "memory"
    );
    while (1) {}
}

static void print_kv(const char *prefix, int32_t val) {
    char buf[64];
    int idx = 0;
    while (prefix[idx] != '\0' && idx < 40) {
        buf[idx] = prefix[idx];
        idx++;
    }
    if (val < 0) {
        buf[idx++] = '-';
        val = -val;
    }
    char digits[16];
    int d = 0;
    if (val == 0) digits[d++] = '0';
    while (val > 0 && d < 15) {
        digits[d++] = (char)('0' + (val % 10));
        val /= 10;
    }
    while (d > 0) buf[idx++] = digits[--d];
    buf[idx++] = '\n';
    buf[idx] = '\0';
    semihost_write0(buf);
}

int main(void) {
    BiquadFilterState filt;
    RPeakDetectorState rdet;
    FaultDetectorState fdet;

    ECG_Filter_Init(&filt);
    RPeak_Init(&rdet);
    Fault_Init(&fdet);

    uint32_t peaks[MAX_PEAKS];
    uint32_t total_peaks = 0U;

    uint32_t tp = 0U, fp = 0U, fn = 0U, tn = 0U;
    uint32_t first_detect_latency = 0U;
    uint8_t seen_fault = 0U;

    float current_ecg_hr = 0.0f;
    float last_valid_hr = 0.0f;

    for (uint32_t w = 0U; w < (DATASET_SIZE / WIN_SAMPLES); w++) {
        uint32_t w_start = w * WIN_SAMPLES;
        uint32_t w_peaks[64];
        uint32_t w_pcount = 0U;
        uint32_t w_ecg_faults = 0U;

        for (uint32_t i = 0U; i < WIN_SAMPLES; i++) {
            uint32_t n = w_start + i;
            float y = ECG_Filter_Process(&filt, ecg_data[n]);
            if (RPeak_Process(&rdet, y, n)) {
                if (total_peaks < MAX_PEAKS) peaks[total_peaks++] = n;
                if (w_pcount < 64U) w_peaks[w_pcount++] = n;
                /* Update rolling 4-beat ECG HR immediately on every R-peak */
                if (total_peaks >= 3U) {
                    uint32_t k = (total_peaks >= 5U) ? 5U : total_peaks;
                    current_ecg_hr = HR_Estimate_RR_Filtered(&peaks[total_peaks - k], k, SAMPLE_RATE, current_ecg_hr);
                }
            }

            FaultCode fc = Fault_Detect_Full(&fdet, dt_ms[n], ecg_data[n],
                                             sensor_hr[n], spo2_data[n], current_ecg_hr);
            uint8_t pred = (fc != FAULT_NONE) ? 1U : 0U;
            uint8_t gt = fault_ground_truth[n];

            if (gt == 1U && !seen_fault) {
                seen_fault = 1U;
                first_detect_latency = (pred == 1U) ? 0U : 999U;
            } else if (seen_fault && first_detect_latency == 999U && pred == 1U) {
                first_detect_latency = n;
            }

            /* Only count ECG-corrupting faults toward ECG HR hold threshold */
            if (fc == FAULT_MISSING_SAMPLE || fc == FAULT_COMM_BURST || fc == FAULT_OUT_OF_RANGE) {
                w_ecg_faults++;
            }
            if (pred == 1U && gt == 1U) tp++;
            else if (pred == 1U && gt == 0U) fp++;
            else if (pred == 0U && gt == 1U) fn++;
            else tn++;
        }

        float hr_raw = HR_Estimate_Raw(w_peaks, w_pcount, SAMPLE_RATE);
        if (last_valid_hr == 0.0f && hr_raw > 0.0f) last_valid_hr = hr_raw;

        float hr_rr  = HR_Estimate_RR_Filtered(w_peaks, w_pcount, SAMPLE_RATE, last_valid_hr);
        float sqi = 0.0f;
        float hr_mit = HR_Estimate_Mitigated(w_peaks, w_pcount, SAMPLE_RATE,
                                             w_ecg_faults, WIN_SAMPLES, last_valid_hr, &sqi);

        if (sqi >= 60.0f && hr_mit > 0.0f) last_valid_hr = hr_mit;

        print_kv("HR_RAW x100 = ", (int32_t)(hr_raw * 100.0f + 0.5f));
        print_kv("HR_RR  x100 = ", (int32_t)(hr_rr  * 100.0f + 0.5f));
        print_kv("HR_MIT x100 = ", (int32_t)(hr_mit * 100.0f + 0.5f));
        print_kv("SQI    x100 = ", (int32_t)(sqi    * 100.0f + 0.5f));
    }

    print_kv("Detected R-peaks: ", (int32_t)total_peaks);
    print_kv("TP: ", (int32_t)tp);
    print_kv("FP: ", (int32_t)fp);
    print_kv("FN: ", (int32_t)fn);
    print_kv("TN: ", (int32_t)tn);
    print_kv("LatencySamples: ", (int32_t)first_detect_latency);

    semihost_exit();
    return 0;
}
