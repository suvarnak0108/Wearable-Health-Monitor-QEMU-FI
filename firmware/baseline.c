
#include <stdio.h>

int main()
{
    FILE *fp;
    char line[200];

    double time, ecg;
    int heart_rate, spo2;

    int total = 0;
    int valid = 0;
    int invalid = 0;

    fp = fopen("datasets/normal_data.csv", "r");

    if (fp == NULL)
    {
        printf("Dataset not found!\n");
        return 1;
    }

    // Skip CSV header
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp))
    {
        if (sscanf(line, "%lf,%lf,%d,%d",
                   &time, &ecg, &heart_rate, &spo2) == 4)
        {
            total++;

            if (heart_rate >= 30 && heart_rate <= 200 &&
                spo2 >= 70 && spo2 <= 100)
            {
                valid++;
            }
            else
            {
                invalid++;
            }
        }
    }

    fclose(fp);

    printf("\n--- Baseline Monitoring Results ---\n");
    printf("Total readings : %d\n", total);
    printf("Valid readings : %d\n", valid);
    printf("Invalid readings: %d\n", invalid);

    return 0;
}
