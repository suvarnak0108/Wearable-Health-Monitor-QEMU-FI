#include "wearable_monitor.h"

static void sort_floats(float *arr, uint32_t n) {
    for (uint32_t i = 1U; i < n; i++) {
        float key = arr[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

float HR_Estimate_Raw(const uint32_t *peaks, uint32_t count, float fs) {
    if (count < 2U || fs <= 0.0f) return 0.0f;
    float rr[64];
    uint32_t n = (count - 1U > 64U) ? 64U : (count - 1U);
    for (uint32_t i = 0U; i < n; i++) {
        rr[i] = (float)(peaks[i + 1U] - peaks[i]);
    }
    sort_floats(rr, n);
    float med = (n % 2U == 1U) ? rr[n / 2U] : 0.5f * (rr[n / 2U - 1U] + rr[n / 2U]);
    return (med > 0.0f) ? (60.0f * fs / med) : 0.0f;
}

float HR_Estimate_RR_Filtered(const uint32_t *peaks, uint32_t count, float fs, float fallback_hr) {
    if (count < 2U || fs <= 0.0f) return fallback_hr;
    float rr[64];
    uint32_t valid = 0U;
    float min_rr = 0.30f * fs; /* 200 BPM */
    float max_rr = 1.50f * fs; /* 40 BPM */

    for (uint32_t i = 0U; i + 1U < count && valid < 64U; i++) {
        float d = (float)(peaks[i + 1U] - peaks[i]);
        if (d >= min_rr && d <= max_rr) {
            rr[valid++] = d;
        }
    }
    if (valid < 2U) return fallback_hr;
    sort_floats(rr, valid);
    float med = (valid % 2U == 1U) ? rr[valid / 2U] : 0.5f * (rr[valid / 2U - 1U] + rr[valid / 2U]);
    return (med > 0.0f) ? (60.0f * fs / med) : fallback_hr;
}

float HR_Estimate_Mitigated(const uint32_t *peaks, uint32_t count, float fs,
                            uint32_t fault_samples, uint32_t win_samples,
                            float last_valid_hr, float *out_sqi) {
    float rho_f = (win_samples > 0U) ? ((float)fault_samples / (float)win_samples) : 1.0f;
    if (count < 2U || fs <= 0.0f) {
        *out_sqi = 0.0f;
        return last_valid_hr;
    }

    float rr[64];
    uint32_t raw_rr = count - 1U;
    uint32_t valid = 0U;
    float min_rr = 0.30f * fs;
    float max_rr = 1.50f * fs;

    for (uint32_t i = 0U; i < raw_rr && valid < 64U; i++) {
        float d = (float)(peaks[i + 1U] - peaks[i]);
        if (d >= min_rr && d <= max_rr) {
            rr[valid++] = d;
        }
    }

    float ratio = (raw_rr > 0U) ? ((float)valid / (float)raw_rr) : 0.0f;
    float reg_factor = 1.0f;
    if (valid >= 2U) {
        sort_floats(rr, valid);
        float med = (valid % 2U == 1U) ? rr[valid / 2U] : 0.5f * (rr[valid / 2U - 1U] + rr[valid / 2U]);
        float span = rr[valid - 1U] - rr[0U];
        float rel_span = (med > 0.0f) ? (span / med) : 1.0f;
        /* Penalize SQI proportionally to beat-to-beat RR spread (ectopy, T-wave double triggers, motion spikes) */
        reg_factor = 1.0f - 0.65f * rel_span;
        if (reg_factor < 0.15f) reg_factor = 0.15f;
    } else {
        reg_factor = 0.0f;
    }

    *out_sqi = 100.0f * (1.0f - rho_f) * ratio * reg_factor;

    /* Engage hold-last-HR when RR intervals are insufficient, fault density > 25%, or SQI < 50% */
    if (valid < 2U || rho_f > 0.25f || *out_sqi < 50.0f) {
        return last_valid_hr;
    }
    float med = (valid % 2U == 1U) ? rr[valid / 2U] : 0.5f * (rr[valid / 2U - 1U] + rr[valid / 2U]);
    return (med > 0.0f) ? (60.0f * fs / med) : last_valid_hr;
}
