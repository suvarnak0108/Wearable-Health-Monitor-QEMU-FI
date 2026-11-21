#include "wearable_monitor.h"

/* 300 ms refractory period @ 100 Hz -> supports up to 200 BPM */
#define REFRACTORY_SAMPLES 30U
#define MIN_ENERGY_FLOOR   0.012f

void RPeak_Init(RPeakDetectorState *st) {
    st->spki = 0.20f;
    st->npki = 0.01f;
    st->threshold = 0.086f;
    st->last_peak_idx = 0U;
    st->prev_e1 = 0.0f;
    st->prev_e2 = 0.0f;
}

uint8_t RPeak_Process(RPeakDetectorState *st, float y_filt, uint32_t idx) {
    /* Full-wave quadratic energy captures both upright and inverted PVCs */
    float e = y_filt * y_filt;
    uint8_t detected = 0U;

    if (idx >= 2U && e < st->prev_e1 && st->prev_e1 >= st->prev_e2) {
        uint32_t cand = idx - 1U;
        float e_peak = st->prev_e1;
        uint32_t dt = cand - st->last_peak_idx;

        if ((st->last_peak_idx == 0U || dt >= REFRACTORY_SAMPLES) && e_peak > st->threshold) {
            st->spki = 0.125f * e_peak + 0.875f * st->spki;
            float th = st->npki + 0.40f * (st->spki - st->npki);
            st->threshold = (th < MIN_ENERGY_FLOOR) ? MIN_ENERGY_FLOOR : th;
            st->last_peak_idx = cand;
            detected = 1U;
        } else if (e_peak > 0.0005f) {
            st->npki = 0.125f * e_peak + 0.875f * st->npki;
            float th = st->npki + 0.40f * (st->spki - st->npki);
            st->threshold = (th < MIN_ENERGY_FLOOR) ? MIN_ENERGY_FLOOR : th;
        }
    }

    st->prev_e2 = st->prev_e1;
    st->prev_e1 = e;
    return detected;
}
