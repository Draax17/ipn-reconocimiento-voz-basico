"""
Módulo para extracción de características MFCC desde cero.
Implementación sin librerías externas de audio (solo NumPy y SciPy).
"""
import numpy as np
from scipy.fftpack import dct
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX,
    INCLUDE_DELTA, INCLUDE_DELTA_DELTA
)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """
    Convierte frecuencia en Hz a escala Mel.
    Fórmula: mel = 2595 * log10(1 + hz/700)
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """
    Convierte escala Mel a frecuencia en Hz.
    Fórmula: hz = 700 * (10^(mel/2595) - 1)
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 0.0,
    fmax: float = None
) -> np.ndarray:
    """
    Crea un banco de filtros triangulares en escala Mel.

    Args:
        n_filters: Número de filtros Mel
        n_fft: Tamaño de la FFT
        sr: Frecuencia de muestreo
        fmin: Frecuencia mínima en Hz
        fmax: Frecuencia máxima en Hz

    Returns:
        Matriz de filtros (n_filters, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = sr / 2.0

    # Convertir frecuencias límite a Mel
    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]

    # Crear puntos equiespaciados en escala Mel
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)

    # Convertir de vuelta a Hz
    hz_points = mel_to_hz(mel_points)

    # Convertir a índices de bins de FFT
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Crear banco de filtros
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_bins))

    for i in range(n_filters):
        # Puntos del filtro triangular
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Pendiente ascendente
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Pendiente descendente
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def apply_preemphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Aplica filtro de pre-énfasis para realzar altas frecuencias.
    y[n] = x[n] - coef * x[n-1]
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """
    Divide la señal en frames con overlap.

    Args:
        signal: Señal de audio
        frame_length: Longitud de cada frame
        hop_length: Salto entre frames

    Returns:
        Matriz de frames (n_frames, frame_length)
    """
    signal_length = len(signal)

    # Calcular número de frames
    n_frames = 1 + (signal_length - frame_length) // hop_length

    if n_frames < 1:
        # Si la señal es muy corta, pad con ceros
        signal = np.pad(signal, (0, frame_length - signal_length))
        n_frames = 1

    # Crear matriz de frames
    frames = np.zeros((n_frames, frame_length))

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(signal):
            frames[i] = signal[start:end]
        else:
            # Pad el último frame si es necesario
            frames[i, :len(signal) - start] = signal[start:]

    return frames


def apply_window(frames: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    """
    Aplica ventana a los frames.

    Args:
        frames: Matriz de frames
        window_type: Tipo de ventana ('hamming', 'hann', 'rectangular')

    Returns:
        Frames con ventana aplicada
    """
    frame_length = frames.shape[1]

    if window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))
    elif window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1)))
    else:  # rectangular
        window = np.ones(frame_length)

    return frames * window


def compute_power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """
    Calcula el espectro de potencia de los frames.

    Args:
        frames: Matriz de frames con ventana aplicada
        n_fft: Tamaño de la FFT

    Returns:
        Espectro de potencia (n_frames, n_fft//2 + 1)
    """
    # Aplicar FFT
    fft_result = np.fft.rfft(frames, n=n_fft)

    # Calcular espectro de potencia (magnitud al cuadrado)
    power_spectrum = (np.abs(fft_result) ** 2) / n_fft

    return power_spectrum


def apply_mel_filterbank(
    power_spectrum: np.ndarray,
    filterbank: np.ndarray
) -> np.ndarray:
    """
    Aplica el banco de filtros Mel al espectro de potencia.

    Args:
        power_spectrum: Espectro de potencia
        filterbank: Banco de filtros Mel

    Returns:
        Energía por filtro Mel (n_frames, n_filters)
    """
    # Multiplicación matricial: (n_frames, n_bins) @ (n_bins, n_filters)
    mel_spectrum = np.dot(power_spectrum, filterbank.T)

    # Evitar log(0) agregando pequeño epsilon
    mel_spectrum = np.where(mel_spectrum == 0, np.finfo(float).eps, mel_spectrum)

    return mel_spectrum


def compute_mfcc(
    mel_spectrum: np.ndarray,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    Calcula los MFCCs aplicando DCT al log del espectro Mel.

    Args:
        mel_spectrum: Espectro Mel
        n_mfcc: Número de coeficientes a extraer

    Returns:
        MFCCs (n_frames, n_mfcc)
    """
    # Aplicar logaritmo
    log_mel = np.log(mel_spectrum)

    # Aplicar DCT tipo II
    mfccs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfccs


def compute_deltas(
    features: np.ndarray,
    width: int = 9
) -> np.ndarray:
    """
    Calcula las derivadas temporales (deltas) de las características.
    Usa regresión lineal local.

    Args:
        features: Matriz de características (n_features, n_frames)
        width: Ancho de la ventana para el cálculo (debe ser impar)

    Returns:
        Matriz de deltas con la misma forma (n_features, n_frames)
    """
    # Asegurar que width sea impar
    if width % 2 == 0:
        width += 1

    half_width = width // 2

    # Guardar forma original
    original_shape = features.shape

    # Determinar si necesitamos transponer
    # Asumimos que viene como (n_features, n_frames) donde n_features < n_frames
    needs_transpose = features.shape[0] < features.shape[1]

    if needs_transpose:
        # Transponer para tener (n_frames, n_features)
        features = features.T

    n_frames, n_features = features.shape

    # Pad temporal con replicación de bordes
    padded = np.pad(features, ((half_width, half_width), (0, 0)), mode='edge')

    # Calcular deltas usando regresión
    deltas = np.zeros_like(features)

    # Denominador de la fórmula de regresión
    denominator = 2 * sum(n ** 2 for n in range(1, half_width + 1))

    for t in range(n_frames):
        numerator = np.zeros(n_features)
        for n in range(1, half_width + 1):
            numerator += n * (padded[t + half_width + n] - padded[t + half_width - n])
        deltas[t] = numerator / denominator

    # Restaurar orientación original
    if needs_transpose:
        deltas = deltas.T

    return deltas


def extract_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    preemphasis: bool = True
) -> np.ndarray:
    """
    Extrae coeficientes MFCC de una señal de audio.
    Implementación desde cero sin librosa.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_mfcc: Número de coeficientes MFCC
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames
        n_mels: Número de filtros mel
        fmin: Frecuencia mínima
        fmax: Frecuencia máxima
        preemphasis: Aplicar pre-énfasis

    Returns:
        Array de MFCCs con forma (n_mfcc, n_frames)
    """
    # 1. Pre-énfasis
    if preemphasis:
        audio = apply_preemphasis(audio)

    # 2. Dividir en frames
    frames = frame_signal(audio, n_fft, hop_length)

    # 3. Aplicar ventana Hamming
    frames = apply_window(frames, 'hamming')

    # 4. Calcular espectro de potencia
    power_spectrum = compute_power_spectrum(frames, n_fft)

    # 5. Crear y aplicar banco de filtros Mel
    filterbank = create_mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
    mel_spectrum = apply_mel_filterbank(power_spectrum, filterbank)

    # 6. Calcular MFCCs
    mfccs = compute_mfcc(mel_spectrum, n_mfcc)

    # Transponer para tener forma (n_mfcc, n_frames)
    return mfccs.T


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
    # Pre-énfasis
    audio = apply_preemphasis(audio)

    # Frames y ventana
    frames = frame_signal(audio, n_fft, hop_length)
    frames = apply_window(frames, 'hamming')

    # Espectro de potencia
    power_spectrum = compute_power_spectrum(frames, n_fft)

    # Banco de filtros Mel
    filterbank = create_mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
    mel_spectrum = apply_mel_filterbank(power_spectrum, filterbank)

    # Log
    log_mel = 10 * np.log10(mel_spectrum + 1e-10)

    return log_mel.T


def extract_additional_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> dict:
    """
    Extrae características adicionales útiles para reconocimiento de voz.
    Implementación desde cero.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames

    Returns:
        Diccionario con características adicionales
    """
    features = {}

    # Dividir en frames
    frames = frame_signal(audio, n_fft, hop_length)

    # Zero Crossing Rate
    zcr = np.zeros(frames.shape[0])
    for i, frame in enumerate(frames):
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr[i] = np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
    features['zcr'] = zcr

    # RMS Energy
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    features['rms'] = rms

    # Spectral Centroid
    frames_windowed = apply_window(frames, 'hamming')
    fft_result = np.abs(np.fft.rfft(frames_windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    spectral_centroid = np.zeros(frames.shape[0])
    for i, spectrum in enumerate(fft_result):
        if np.sum(spectrum) > 0:
            spectral_centroid[i] = np.sum(freqs * spectrum) / np.sum(spectrum)
    features['spectral_centroid'] = spectral_centroid

    # Spectral Rolloff (frecuencia bajo la cual está el 85% de la energía)
    spectral_rolloff = np.zeros(frames.shape[0])
    for i, spectrum in enumerate(fft_result):
        cumsum = np.cumsum(spectrum)
        if cumsum[-1] > 0:
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            spectral_rolloff[i] = freqs[min(rolloff_idx, len(freqs) - 1)]
    features['spectral_rolloff'] = spectral_rolloff

    # Spectral Bandwidth
    spectral_bandwidth = np.zeros(frames.shape[0])
    for i, spectrum in enumerate(fft_result):
        if np.sum(spectrum) > 0:
            centroid = spectral_centroid[i]
            spectral_bandwidth[i] = np.sqrt(
                np.sum(spectrum * (freqs - centroid) ** 2) / np.sum(spectrum)
            )
    features['spectral_bandwidth'] = spectral_bandwidth

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
                # Interpolar o recortar
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, feat.shape[1])
                x_new = np.linspace(0, 1, mfcc_features.shape[1])
                f = interp1d(x_old, feat, axis=1, fill_value='extrapolate')
                feat = f(x_new)
            all_features.append(feat)

        return np.vstack(all_features)

    return mfcc_features


def aggregate_features(features: np.ndarray) -> np.ndarray:
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


# Test rápido si se ejecuta directamente
if __name__ == "__main__":
    print("Test de extracción de MFCCs desde cero")
    print("=" * 50)

    # Crear señal de prueba (tono de 440 Hz)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    print(f"Audio de prueba: {len(audio)} muestras, {sr} Hz")

    # Extraer MFCCs
    mfccs = extract_mfcc(audio, sr)
    print(f"MFCCs: {mfccs.shape}")

    # Extraer con deltas
    mfccs_deltas = extract_mfcc_with_deltas(audio, sr)
    print(f"MFCCs + deltas: {mfccs_deltas.shape}")

    # Agregar
    aggregated = aggregate_features(mfccs_deltas)
    print(f"Features agregadas: {aggregated.shape}")

    print("\n¡Test completado!")
