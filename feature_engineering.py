import numpy as np
from scipy import stats
from scipy.fft import fft

def extract_statistical_features(X):
    return np.column_stack([
        np.mean(X, axis=1), np.std(X, axis=1), np.var(X, axis=1),
        np.median(X, axis=1), np.min(X, axis=1), np.max(X, axis=1),
        np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
        stats.skew(X, axis=1), stats.kurtosis(X, axis=1)
    ])

def extract_frequency_features(X):
    fft_vals = np.abs(fft(X, axis=1))
    n_half = X.shape[1] // 2
    fft_vals = fft_vals[:, :n_half]
    return np.column_stack([
        np.sum(fft_vals, axis=1), np.mean(fft_vals, axis=1),
        np.std(fft_vals, axis=1), np.max(fft_vals, axis=1)
    ])

def extract_temporal_features(X):
    zero_crossings = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1)
    first_diff = np.diff(X, axis=1)
    return np.column_stack([
        zero_crossings,
        np.mean(first_diff, axis=1),
        np.std(first_diff, axis=1),
        np.sum(X**2, axis=1)
    ])

def extract_all_features(X):
    stat_features = extract_statistical_features(X)
    freq_features = extract_frequency_features(X)
    temp_features = extract_temporal_features(X)
    return np.column_stack([stat_features, freq_features, temp_features])
