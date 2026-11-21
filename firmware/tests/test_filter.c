#include <stdio.h>
#include "ecg_filter.h"

int main(void)
{
    ECGFilter filter;

    ECG_Filter_Init(&filter);

    printf("ECG Filter Test Started\n");

    for (int i = 0; i < 20; i++)
    {
        float input = (i == 0) ? 1.0f : 0.0f;

        float output = ECG_Filter_Process(
            &filter, input
        );

        printf(
            "Sample %d: Input = %.4f, Output = %.6f\n",
            i, input, output
        );
    }

    printf("Filter test completed.\n");

    return 0;
}
