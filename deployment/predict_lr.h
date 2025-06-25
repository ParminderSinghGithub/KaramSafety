#ifndef PREDICT_LR_H
#define PREDICT_LR_H

#include "../model_lr.h"
#include <math.h>

// === Stage 1: Scale the 40 aggregated features ===
void scale_aggregated(float input40[], float output40[]) {
    for (int i = 0; i < AGG_FEATURES; ++i) {
        output40[i] = (input40[i] - scaler_agg_mean[i]) / scaler_agg_std[i];
    }
}

// === Stage 2a: Statistical Features (10 features)
void extract_statistical_features(float input[10][51], float output[10]) {
    for (int i = 0; i < 10; ++i) {
        float sum = 0, sq_sum = 0;
        float min_v = input[i][0], max_v = input[i][0];
        float percentile25 = 0, percentile75 = 0;
        float values[51];

        for (int j = 0; j < 51; ++j) {
            float val = input[i][j];
            values[j] = val;
            sum += val;
            sq_sum += val * val;
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
        }

        float mean = sum / 51.0f;
        float var = (sq_sum / 51.0f) - (mean * mean);
        float std = sqrt(var);

        // Sort for percentiles and median
        for (int a = 0; a < 50; ++a) {
            for (int b = a + 1; b < 51; ++b) {
                if (values[a] > values[b]) {
                    float tmp = values[a];
                    values[a] = values[b];
                    values[b] = tmp;
                }
            }
        }

        float median = values[25];
        percentile25 = values[12];
        percentile75 = values[38];

        // Skewness and kurtosis (simplified moment calculations)
        float skew = 0, kurt = 0;
        for (int j = 0; j < 51; ++j) {
            float diff = input[i][j] - mean;
            skew += pow(diff, 3);
            kurt += pow(diff, 4);
        }
        skew = skew / (51.0f * pow(std, 3));
        kurt = kurt / (51.0f * pow(std, 4));

        // Output 10 features
        output[i * 10 + 0] = mean;
        output[i * 10 + 1] = std;
        output[i * 10 + 2] = var;
        output[i * 10 + 3] = median;
        output[i * 10 + 4] = min_v;
        output[i * 10 + 5] = max_v;
        output[i * 10 + 6] = percentile25;
        output[i * 10 + 7] = percentile75;
        output[i * 10 + 8] = skew;
        output[i * 10 + 9] = kurt;
    }
}

// === Stage 2b: Frequency Features (4 features)
void extract_frequency_features(float input[10][51], float output[4]) {
    float fft_real[51];
    float fft_imag[51];
    float magnitude[51];
    float sum = 0, max_mag = 0, sq_sum = 0;

    for (int i = 0; i < 10; ++i) {
        for (int k = 0; k < 51; ++k) {
            fft_real[k] = 0;
            fft_imag[k] = 0;
            for (int n = 0; n < 51; ++n) {
                float angle = 2 * M_PI * k * n / 51.0;
                fft_real[k] += input[i][n] * cos(angle);
                fft_imag[k] -= input[i][n] * sin(angle);
            }
            magnitude[k] = sqrt(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]);
        }

        for (int j = 0; j < 25; ++j) {
            float m = magnitude[j];
            sum += m;
            sq_sum += m * m;
            if (m > max_mag) max_mag = m;
        }
    }

    float mean = sum / (10.0f * 25);
    float std = sqrt(sq_sum / (10.0f * 25) - mean * mean);

    output[0] = sum;
    output[1] = mean;
    output[2] = std;
    output[3] = max_mag;
}

// === Stage 2c: Temporal Features (4 features)
void extract_temporal_features(float input[10][51], float output[4]) {
    int zero_crossings = 0;
    float diff_sum = 0, diff_sq_sum = 0, energy = 0;

    for (int i = 0; i < 10; ++i) {
        for (int j = 1; j < 51; ++j) {
            float prev = input[i][j - 1];
            float curr = input[i][j];
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) zero_crossings++;
            float diff = curr - prev;
            diff_sum += diff;
            diff_sq_sum += diff * diff;
            energy += curr * curr;
        }
    }

    output[0] = zero_crossings;
    output[1] = diff_sum / (10 * 50);
    output[2] = sqrt(diff_sq_sum / (10 * 50));
    output[3] = energy;
}

// === Stage 2: Combine all features ===
void extract_all_features(float input[10][51], float output[58]) {
    float stat[100];
    float freq[4];
    float temp[4];

    extract_statistical_features(input, stat);
    extract_frequency_features(input, freq);
    extract_temporal_features(input, temp);

    for (int i = 0; i < 100; ++i) output[i] = stat[i];
    for (int i = 0; i < 4; ++i) output[100 + i] = freq[i];
    for (int i = 0; i < 4; ++i) output[104 + i] = temp[i];
}

// === Stage 3: Feature selection from 58 to 47 ===
void select_features(float input58[], float selected47[]) {
    for (int i = 0; i < NUM_SELECTED; ++i) {
        selected47[i] = input58[selected_indices[i]];
    }
}

// === Stage 4: Final scaling before model ===
void scale_final(float input47[], float output47[]) {
    for (int i = 0; i < NUM_FEATURES; ++i) {
        output47[i] = (input47[i] - scaler_final_mean[i]) / scaler_final_std[i];
    }
}

// === Stage 5: Prediction ===
int predict_class(float input47[]) {
    float logits[NUM_CLASSES];
    float max_val = -1e9;
    int max_idx = 0;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        logits[i] = bias[i];
        for (int j = 0; j < NUM_FEATURES; ++j) {
            logits[i] += weights[i][j] * input47[j];
        }
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

#endif // PREDICT_LR_H
