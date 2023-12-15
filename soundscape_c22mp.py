from c22mp import get_feature_profiles, feature_search_regression, left_c22_mp
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_spec(spectrogram):
    spectrogram = gaussian_filter(spectrogram, sigma=2, axes=0)
    spectrogram = spectrogram - spectrogram.mean(1, keepdims=True)
    spectrogram = spectrogram / spectrogram.std(1, keepdims=True)
    return spectrogram

def get_left_c22mp(ts, m):
    start_idx = m * 2
    fpn = get_feature_profiles(ts, m)
    weights = feature_search_regression(fpn, ts[m - 1 :])

    return left_c22_mp(
        X=fpn,
        start_idx=start_idx,
        win=m,
        dynamic_bsf_update=True,
        weights=weights,
    )[0]

def get_boolean_region(v, smoothing, sigmas):
    v = np.convolve(v, np.ones(smoothing) / smoothing, mode="same")
    v -= v.mean()
    v /= v.std()
    v = (v > sigmas).astype(bool)
    return v.tolist()

def get_frequencies(spectrogram):
    return spectrogram.argmax(0)

def detect_anomalies(audio_file, m=300, smoothing=200, sigmas=1):
    wav, sr = librosa.load(audio_file, sr=None)
    spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=128
    )
    spectrogram = librosa.power_to_db(spectrogram)
    
    # preprocess
    smooth_spectrogram = smooth_spec(spectrogram)
    ts = get_frequencies(smooth_spectrogram)
    
    # getting anomalies with c22mp
    v = get_left_c22mp(ts, m)

    # getting boolean region
    b_region = get_boolean_region(v, smoothing, sigmas)

    return b_region

