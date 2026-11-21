#include "wearable_monitor.h"

/* 3-section 2nd-order Butterworth bandpass (5-15 Hz @ fs = 100 Hz) */
static const float b0[3] = {0.06745527f, 1.0f, 1.0f};
static const float b1[3] = {0.0f, 2.0f, -2.0f};
static const float b2[3] = {-0.06745527f, 1.0f, 1.0f};
static const float a1[3] = {-1.14298050f, -1.04859958f, -1.32091343f};
static const float a2[3] = {0.41280160f, 0.29614036f, 0.63273879f};

void ECG_Filter_Init(BiquadFilterState *st) {
    for (int i = 0; i < 3; i++) {
        st->z1[i] = 0.0f;
        st->z2[i] = 0.0f;
    }
}

float ECG_Filter_Process(BiquadFilterState *st, float x) {
    float v = x;
    for (int k = 0; k < 3; k++) {
        float w = v - a1[k] * st->z1[k] - a2[k] * st->z2[k];
        float y = b0[k] * w + b1[k] * st->z1[k] + b2[k] * st->z2[k];
        st->z2[k] = st->z1[k];
        st->z1[k] = w;
        v = y;
    }
    return v;
}
