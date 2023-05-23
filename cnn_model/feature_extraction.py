import librosa
import numpy as np
import pandas as pd
from scipy import signal
import pysptk
import os
from tqdm import tqdm
import noisereduce as nr

def spectral_flux(audio_data, window_size, hop_size):
    frames = librosa.util.frame(audio_data, frame_length=window_size, hop_length=hop_size).T
    window = np.hanning(window_size)
    normalized_frames = frames * window
    power_spectra = np.abs(np.fft.rfft(normalized_frames, axis=1))
    return np.sum(np.diff(power_spectra, axis=0)**2, axis=1)

def extract_features_from_video(audio_file_path, video_id):
    # load the audio file
    audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)

    audio_data = nr.reduce_noise(y=audio_data, sr=sampling_rate)

    if np.isnan(audio_data).any() or np.isinf(audio_data).any():
        audio_data = np.nan_to_num(audio_data)

    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)[0]
    spectral_centroids_mean = np.mean(spectral_centroids)
    spectral_centroids_std = np.std(spectral_centroids)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data + 0.01, sr=sampling_rate)[0]
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)

    spectral_fluxes = spectral_flux(audio_data, window_size=2048, hop_size=1024)
    spectral_fluxes_mean = np.mean(spectral_fluxes)
    spectral_fluxes_std = np.std(spectral_fluxes)
    
    rms = librosa.feature.rms(y=audio_data)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zero_crossings = librosa.zero_crossings(y=audio_data, pad=False)
    zero_crossings_mean = np.mean(zero_crossings)
    zero_crossings_std = np.std(zero_crossings)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate)
    mfccs_mean = np.mean(mfccs)
    mfccs_std = np.std(mfccs)

    lpcs = signal.lfilter([0] + -1 * np.r_[1, signal.lfilter([1, 0.63], 1, pysptk.lpc(audio_data, 10))], \
                          1, audio_data)
    lpcs_mean = np.mean(lpcs)
    lpcs_std = np.std(lpcs)

    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_std = np.std(chroma_stft)

    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sampling_rate)
    spectral_contrast_mean = np.mean(spectral_contrast)
    spectral_contrast_std = np.std(spectral_contrast)

    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sampling_rate)
    tonnetz_mean = np.mean(tonnetz)
    tonnetz_std = np.std(tonnetz)

    return (
        pd.DataFrame(
            [
                [
                    video_id,
                    spectral_centroids_std,
                    spectral_rolloff_std,
                    spectral_fluxes_std,
                    rms_std,
                    zero_crossings_std,
                    mfccs_std,
                    lpcs_std,
                    spectral_centroids_mean,
                    spectral_rolloff_mean,
                    spectral_fluxes_mean,
                    rms_mean,
                    zero_crossings_mean,
                    mfccs_mean,
                    lpcs_mean,
                    chroma_stft_mean,
                    chroma_stft_std,
                    spectral_contrast_mean,
                    spectral_contrast_std,
                    tonnetz_mean,
                    tonnetz_std,
                ]
            ],
            columns=[
                'video_id',
                'spectral_centroids_std',
                'spectral_rolloff_std',
                'spectral_fluxes_std',
                'rms_std',
                'zero_crossings_std',
                'mfccs_std',
                'lpcs_std',
                'spectral_centroids_mean',
                'spectral_rolloff_mean',
                'spectral_fluxes_mean',
                'rms_mean',
                'zero_crossings_mean',
                'mfccs_mean',
                'lpcs_mean',
                'chroma_stft_mean',
                'chroma_stft_std',
                'spectral_contrast_mean',
                'spectral_contrast_std',
                'tonnetz_mean',
                'tonnetz_std',
            ]
        )
    )

def collect_features(df):
    features_path = os.path.join('data', 'features.csv')
    if os.path.exists(features_path):
        new_df = pd.read_csv(features_path)
    else:
        new_df = pd.DataFrame()
        for row in tqdm(df.itertuples(), total=len(df), desc=f"Collecting audio features"):
            # load audio clip
            audio_path = os.path.join("..", row.audio_file)
            new_df = pd.concat([new_df, extract_features_from_video(audio_path, row.video_id)], ignore_index=True)
        new_df.to_csv(features_path, index=False)

    return new_df