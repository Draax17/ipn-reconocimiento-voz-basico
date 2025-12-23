"""
Módulo para extracción de características MFCC.
Implementa extracción de MFCCs, deltas y otras características acústicas.
"""
import numpy as np
from typing import Optional, Tuple
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
    INCLUDE_DELTA, INCLUDE_DELTA_DELTA
)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX
) -> np.ndarray:
    """
    Extrae coeficientes MFCC de una señal de audio.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames
        n_mels: Número de filtros mel
        fmin: Frecuencia mínima
        fmax: Frecuencia máxima

    Returns:
        Array de MFCCs con forma (n_mfcc, n_frames)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    return mfccs


def compute_deltas(
    features: np.ndarray,
    width: int = 9
) -> np.ndarray:
    """
    Calcula las derivadas temporales (deltas) de las características.

    Args:
        features: Matriz de características (n_features, n_frames)
        width: Ancho de la ventana para el cálculo

    Returns:
        Matriz de deltas con la misma forma
    """
    return librosa.feature.delta(features, width=width)


def extract_mfcc_with_deltas(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    include_delta: bool = INCLUDE_DELTA,
    include_delta_delta: bool = INCLUDE_DELTA_DELTA
) -> np.ndarray:
    """
    Extrae MFCCs junto con sus derivadas primera y segunda.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames
        n_mels: Número de filtros mel
        fmin: Frecuencia mínima
        fmax: Frecuencia máxima
        include_delta: Incluir derivadas de primer orden
        include_delta_delta: Incluir derivadas de segundo orden

    Returns:
        Array de características (n_features, n_frames)
        - Sin deltas: 13 features
        - Con delta: 26 features
        - Con delta-delta: 39 features
    """
    # Extraer MFCCs base
    mfccs = extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length, n_mels, fmin, fmax)

    features = [mfccs]

    # Agregar deltas
    if include_delta:
        delta = compute_deltas(mfccs)
        features.append(delta)

        # Agregar delta-deltas
        if include_delta_delta:
            delta_delta = compute_deltas(delta)
            features.append(delta_delta)

    return np.vstack(features)


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX
) -> np.ndarray:
    """
    Extrae espectrograma log-mel.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames
        n_mels: Número de filtros mel
        fmin: Frecuencia mínima
        fmax: Frecuencia máxima

    Returns:
        Espectrograma log-mel (n_mels, n_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec


def extract_additional_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> dict:
    """
    Extrae características adicionales útiles para reconocimiento de voz.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames

    Returns:
        Diccionario con características adicionales
    """
    features = {}

    # Zero Crossing Rate
    features['zcr'] = librosa.feature.zero_crossing_rate(
        audio, frame_length=n_fft, hop_length=hop_length
    )[0]

    # RMS Energy
    features['rms'] = librosa.feature.rms(
        y=audio, frame_length=n_fft, hop_length=hop_length
    )[0]

    # Spectral Centroid
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Spectral Rolloff
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Spectral Bandwidth
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    return features


def extract_all_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    include_additional: bool = False
) -> np.ndarray:
    """
    Extrae todas las características para un audio.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        include_additional: Incluir ZCR, RMS, etc.

    Returns:
        Matriz de características (n_features, n_frames)
    """
    # MFCCs con deltas
    mfcc_features = extract_mfcc_with_deltas(audio, sr)

    if include_additional:
        additional = extract_additional_features(audio, sr)

        # Apilar todas las características
        all_features = [mfcc_features]
        for name, feat in additional.items():
            # Asegurar que tenga la forma correcta
            if len(feat.shape) == 1:
                feat = feat.reshape(1, -1)
            # Ajustar longitud si es necesario
            if feat.shape[1] != mfcc_features.shape[1]:
                feat = np.resize(feat, (feat.shape[0], mfcc_features.shape[1]))
            all_features.append(feat)

        return np.vstack(all_features)

    return mfcc_features


def aggregate_features(
    features: np.ndarray
) -> np.ndarray:
    """
    Agrega características temporales en un vector fijo.
    Útil para clasificadores que no manejan secuencias.

    Args:
        features: Matriz de características (n_features, n_frames)

    Returns:
        Vector de características agregadas
    """
    aggregated = []

    # Media por característica
    aggregated.extend(np.mean(features, axis=1))

    # Desviación estándar
    aggregated.extend(np.std(features, axis=1))

    # Mínimo y máximo
    aggregated.extend(np.min(features, axis=1))
    aggregated.extend(np.max(features, axis=1))

    # Pendiente (tendencia temporal)
    n_frames = features.shape[1]
    if n_frames > 1:
        x = np.arange(n_frames)
        slopes = []
        for i in range(features.shape[0]):
            slope = np.polyfit(x, features[i], 1)[0]
            slopes.append(slope)
        aggregated.extend(slopes)
    else:
        aggregated.extend(np.zeros(features.shape[0]))

    return np.array(aggregated)
