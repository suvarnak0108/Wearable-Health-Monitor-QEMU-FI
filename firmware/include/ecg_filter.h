#ifndef ECG_FILTER_H
#define ECG_FILTER_H

#define FILTER_SECTIONS 3

typedef struct {
    float x1;
    float x2;
    float y1;
    float y2;
} BiquadState;

typedef struct {
    BiquadState state[FILTER_SECTIONS];
} ECGFilter;

void ECG_Filter_Init(ECGFilter *filter);

float ECG_Filter_Process(
    ECGFilter *filter,
    float input
);

#endif
