"""
Módulo para Voice Activity Detection (VAD).
Detecta segmentos de voz y elimina silencios.
"""
import numpy as np
from typing import List, Tuple, Optional


def compute_energy(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """
    Calcula la energía por frame.

    Args:
        audio: Señal de audio
        frame_length: Longitud del frame en muestras
        hop_length: Salto entre frames

    Returns:
        Array con energía por frame (en dB)
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    energy = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        frame_energy = np.sum(frame ** 2)
        energy[i] = 10 * np.log10(frame_energy + 1e-10)

    return energy


def compute_zcr(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """
    Calcula Zero Crossing Rate por frame.

    Args:
        audio: Señal de audio
        frame_length: Longitud del frame
        hop_length: Salto entre frames

    Returns:
        Array con ZCR por frame
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    zcr = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))

    return zcr


def detect_voice_activity(
    audio: np.ndarray,
    sr: int,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    energy_threshold_db: float = -40.0,
    min_speech_duration_ms: float = 50.0,
    min_silence_duration_ms: float = 100.0
) -> List[Tuple[float, float]]:
    """
    Detecta segmentos de actividad de voz basándose en energía.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        frame_length_ms: Longitud del frame en ms
        hop_length_ms: Salto en ms
        energy_threshold_db: Umbral de energía en dB
        min_speech_duration_ms: Duración mínima de segmento de voz
        min_silence_duration_ms: Duración mínima de silencio

    Returns:
        Lista de tuplas (inicio, fin) en segundos
    """
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)

    energy = compute_energy(audio, frame_length, hop_length)

    # Clasificar frames
    is_speech = energy > energy_threshold_db

    # Encontrar segmentos
    segments = []
    in_speech = False
    start_frame = 0

    for i, speech in enumerate(is_speech):
        if speech and not in_speech:
            start_frame = i
            in_speech = True
        elif not speech and in_speech:
            end_frame = i
            duration_ms = (end_frame - start_frame) * hop_length_ms
            if duration_ms >= min_speech_duration_ms:
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                segments.append((start_time, end_time))
            in_speech = False

    # Si termina en speech
    if in_speech:
        end_frame = len(is_speech)
        duration_ms = (end_frame - start_frame) * hop_length_ms
        if duration_ms >= min_speech_duration_ms:
            start_time = start_frame * hop_length / sr
            end_time = min(end_frame * hop_length / sr, len(audio) / sr)
            segments.append((start_time, end_time))

    # Fusionar segmentos cercanos
    if len(segments) > 1:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            gap_ms = (start - prev_end) * 1000
            if gap_ms < min_silence_duration_ms:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        segments = merged

    return segments


def trim_silence(
    audio: np.ndarray,
    sr: int,
    energy_threshold_db: float = -40.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    margin_ms: float = 50.0
) -> np.ndarray:
    """
    Recorta silencios al inicio y final del audio.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        energy_threshold_db: Umbral de energía
        frame_length_ms: Longitud del frame
        hop_length_ms: Salto entre frames
        margin_ms: Margen a mantener en los bordes

    Returns:
        Audio recortado
    """
    if len(audio) == 0:
        return audio

    segments = detect_voice_activity(
        audio, sr,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        energy_threshold_db=energy_threshold_db
    )

    if not segments:
        return audio

    # Tomar desde el primer segmento hasta el último
    start_time = max(0, segments[0][0] - margin_ms / 1000)
    end_time = min(len(audio) / sr, segments[-1][1] + margin_ms / 1000)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    return audio[start_sample:end_sample]


def get_speech_ratio(
    audio: np.ndarray,
    sr: int,
    energy_threshold_db: float = -40.0
) -> float:
    """
    Calcula el ratio de frames con voz vs total.

    Args:
        audio: Señal de audio
        sr: Frecuencia de muestreo
        energy_threshold_db: Umbral de energía

    Returns:
        Ratio de voz (0-1)
    """
    segments = detect_voice_activity(audio, sr, energy_threshold_db=energy_threshold_db)

    if not segments:
        return 0.0

    total_speech = sum(end - start for start, end in segments)
    total_duration = len(audio) / sr

    return total_speech / total_duration if total_duration > 0 else 0.0
