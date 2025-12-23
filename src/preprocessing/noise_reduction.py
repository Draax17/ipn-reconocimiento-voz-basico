"""
Módulo para reducción de ruido en señales de audio.
"""
import numpy as np
from scipy import signal
from typing import Optional, Tuple


def apply_bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_freq: float = 80.0,
    high_freq: float = 8000.0,
    order: int = 5
) -> np.ndarray:
    """
    Aplica un filtro pasa-banda para eliminar frecuencias fuera del rango del habla.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        low_freq: Frecuencia de corte inferior (Hz)
        high_freq: Frecuencia de corte superior (Hz)
        order: Orden del filtro

    Returns:
        Audio filtrado
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)  # Evitar frecuencia de Nyquist exacta

    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_frames: int = 10,
    n_fft: int = 512,
    hop_length: int = 160,
    alpha: float = 2.0,
    beta: float = 0.01
) -> np.ndarray:
    """
    Reduce ruido usando sustracción espectral.
    Estima el ruido de los primeros frames y lo sustrae del espectro.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        noise_frames: Número de frames iniciales para estimar ruido
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames
        alpha: Factor de sobre-sustracción
        beta: Piso espectral para evitar valores negativos

    Returns:
        Audio con ruido reducido
    """
    # STFT
    f, t, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimar espectro de ruido de los primeros frames
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Sustracción espectral
    magnitude_clean = magnitude - alpha * noise_spectrum

    # Aplicar piso espectral
    magnitude_clean = np.maximum(magnitude_clean, beta * magnitude)

    # Reconstruir señal
    stft_clean = magnitude_clean * np.exp(1j * phase)
    _, audio_clean = signal.istft(stft_clean, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)

    # Ajustar longitud
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    return audio_clean.astype(np.float32)


def wiener_filter(
    audio: np.ndarray,
    sr: int,
    noise_frames: int = 10,
    n_fft: int = 512,
    hop_length: int = 160
) -> np.ndarray:
    """
    Aplica filtro de Wiener para reducción de ruido.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        noise_frames: Frames para estimar ruido
        n_fft: Tamaño de la FFT
        hop_length: Salto entre frames

    Returns:
        Audio filtrado
    """
    # STFT
    f, t, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    power = np.abs(stft) ** 2
    phase = np.angle(stft)

    # Estimar potencia del ruido
    noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)

    # Filtro de Wiener: H = max(1 - noise/signal, 0)
    snr = power / (noise_power + 1e-10)
    wiener_gain = np.maximum(1 - 1/snr, 0)

    # Aplicar ganancia
    magnitude_clean = np.sqrt(power * wiener_gain)

    # Reconstruir
    stft_clean = magnitude_clean * np.exp(1j * phase)
    _, audio_clean = signal.istft(stft_clean, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)

    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    return audio_clean.astype(np.float32)


def reduce_noise(
    audio: np.ndarray,
    sr: int,
    method: str = "spectral",
    **kwargs
) -> np.ndarray:
    """
    Función principal para reducción de ruido.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        method: Método a usar ("spectral", "wiener", "bandpass")
        **kwargs: Argumentos adicionales para el método

    Returns:
        Audio con ruido reducido
    """
    if method == "spectral":
        return spectral_subtraction(audio, sr, **kwargs)
    elif method == "wiener":
        return wiener_filter(audio, sr, **kwargs)
    elif method == "bandpass":
        return apply_bandpass_filter(audio, sr, **kwargs)
    else:
        raise ValueError(f"Método desconocido: {method}")
