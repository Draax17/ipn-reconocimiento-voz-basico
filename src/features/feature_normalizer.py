"""
Módulo para normalización de características.
Implementa CMN (Cepstral Mean Normalization) y CMVN.
"""
import numpy as np
from typing import Optional, Tuple


def apply_cmn(
    features: np.ndarray
) -> np.ndarray:
    """
    Aplica Cepstral Mean Normalization (CMN).
    Resta la media temporal de cada coeficiente.

    Args:
        features: Matriz de características (n_features, n_frames)

    Returns:
        Características normalizadas
    """
    mean = np.mean(features, axis=1, keepdims=True)
    return features - mean


def apply_cmvn(
    features: np.ndarray,
    variance_floor: float = 1e-10
) -> np.ndarray:
    """
    Aplica Cepstral Mean and Variance Normalization (CMVN).
    Normaliza cada coeficiente a media 0 y varianza 1.

    Args:
        features: Matriz de características (n_features, n_frames)
        variance_floor: Valor mínimo de varianza para evitar división por cero

    Returns:
        Características normalizadas
    """
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True)
    std = np.maximum(std, variance_floor)

    return (features - mean) / std


def apply_global_cmvn(
    features: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    variance_floor: float = 1e-10
) -> np.ndarray:
    """
    Aplica CMVN usando estadísticas globales (calculadas sobre todo el dataset).

    Args:
        features: Matriz de características (n_features, n_frames)
        global_mean: Media global por característica
        global_std: Desviación estándar global por característica
        variance_floor: Valor mínimo de varianza

    Returns:
        Características normalizadas
    """
    global_mean = global_mean.reshape(-1, 1)
    global_std = np.maximum(global_std.reshape(-1, 1), variance_floor)

    return (features - global_mean) / global_std


def compute_global_stats(
    all_features: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula estadísticas globales para CMVN sobre un conjunto de características.

    Args:
        all_features: Lista de matrices de características

    Returns:
        Tupla (media_global, std_global)
    """
    # Concatenar todos los frames
    concatenated = np.hstack(all_features)

    global_mean = np.mean(concatenated, axis=1)
    global_std = np.std(concatenated, axis=1)

    return global_mean, global_std


class FeatureNormalizer:
    """
    Clase para normalización de características con estadísticas persistentes.
    """

    def __init__(self, method: str = "cmvn"):
        """
        Args:
            method: Método de normalización ("cmn", "cmvn", "global_cmvn")
        """
        self.method = method
        self.global_mean = None
        self.global_std = None
        self.is_fitted = False

    def fit(self, all_features: list):
        """
        Ajusta el normalizador con las estadísticas del conjunto de entrenamiento.

        Args:
            all_features: Lista de matrices de características
        """
        if self.method == "global_cmvn":
            self.global_mean, self.global_std = compute_global_stats(all_features)
        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Aplica la normalización a las características.

        Args:
            features: Matriz de características

        Returns:
            Características normalizadas
        """
        if self.method == "cmn":
            return apply_cmn(features)
        elif self.method == "cmvn":
            return apply_cmvn(features)
        elif self.method == "global_cmvn":
            if not self.is_fitted:
                raise RuntimeError("Normalizador no ajustado. Llama a fit() primero.")
            return apply_global_cmvn(features, self.global_mean, self.global_std)
        else:
            raise ValueError(f"Método desconocido: {self.method}")

    def fit_transform(self, all_features: list) -> list:
        """
        Ajusta y transforma en un solo paso.

        Args:
            all_features: Lista de matrices de características

        Returns:
            Lista de características normalizadas
        """
        self.fit(all_features)
        return [self.transform(f) for f in all_features]

    def save(self, path: str):
        """Guarda las estadísticas del normalizador."""
        np.savez(
            path,
            method=self.method,
            global_mean=self.global_mean,
            global_std=self.global_std,
            is_fitted=self.is_fitted
        )

    def load(self, path: str):
        """Carga las estadísticas del normalizador."""
        data = np.load(path, allow_pickle=True)
        self.method = str(data['method'])
        self.global_mean = data['global_mean']
        self.global_std = data['global_std']
        self.is_fitted = bool(data['is_fitted'])
