"""
Módulo para normalización de señales de audio.
"""
import numpy as np
from typing import Optional


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = 0.95
) -> np.ndarray:
    """
    Normaliza el audio al pico máximo especificado.

    Args:
        audio: Señal de audio como array numpy
        target_peak: Valor pico objetivo (default: 0.95)

    Returns:
        Audio normalizado
    """
    if len(audio) == 0:
        return audio

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (target_peak / max_val)

    return audio


def normalize_rms(
    audio: np.ndarray,
    target_rms: float = 0.1
) -> np.ndarray:
    """
    Normaliza el audio a un nivel RMS objetivo.
    Útil para hacer que diferentes grabaciones tengan volumen similar.

    Args:
        audio: Señal de audio como array numpy
        target_rms: Nivel RMS objetivo (default: 0.1)

    Returns:
        Audio normalizado por RMS
    """
    if len(audio) == 0:
        return audio

    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        audio = audio * (target_rms / current_rms)

    # Evitar clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    return audio


def apply_preemphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Aplica filtro de pre-énfasis para realzar altas frecuencias.
    y[n] = x[n] - coef * x[n-1]

    Args:
        audio: Señal de audio
        coef: Coeficiente de pre-énfasis (default: 0.97)

    Returns:
        Audio con pre-énfasis aplicado
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remueve el offset DC de la señal.

    Args:
        audio: Señal de audio

    Returns:
        Audio sin offset DC
    """
    return audio - np.mean(audio)


def preprocess_audio(
    audio: np.ndarray,
    remove_dc: bool = True,
    normalize: bool = True,
    preemphasis: bool = True,
    preemphasis_coef: float = 0.97
) -> np.ndarray:
    """
    Aplica pipeline completo de preprocesamiento.

    Args:
        audio: Señal de audio
        remove_dc: Remover offset DC
        normalize: Normalizar amplitud
        preemphasis: Aplicar pre-énfasis
        preemphasis_coef: Coeficiente de pre-énfasis

    Returns:
        Audio preprocesado
    """
    if remove_dc:
        audio = remove_dc_offset(audio)

    if normalize:
        audio = normalize_audio(audio)

    if preemphasis:
        audio = apply_preemphasis(audio, preemphasis_coef)

    return audio
