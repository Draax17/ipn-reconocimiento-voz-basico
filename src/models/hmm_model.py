"""
Módulo para modelo HMM-GMM de reconocimiento de palabras aisladas.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

from hmmlearn import hmm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import HMM_N_STATES, HMM_N_COMPONENTS, HMM_COVARIANCE_TYPE, HMM_N_ITER


class HMMClassifier:
    """
    Clasificador basado en HMM-GMM para reconocimiento de palabras aisladas.
    Entrena un modelo HMM por cada palabra del vocabulario.
    """

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        n_components: int = HMM_N_COMPONENTS,
        covariance_type: str = HMM_COVARIANCE_TYPE,
        n_iter: int = HMM_N_ITER,
        random_state: int = 42
    ):
        """
        Args:
            n_states: Número de estados ocultos por modelo
            n_components: Número de componentes GMM por estado
            covariance_type: Tipo de covarianza ("diag", "full", "spherical")
            n_iter: Número máximo de iteraciones EM
            random_state: Semilla para reproducibilidad
        """
        self.n_states = n_states
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.models: Dict[str, hmm.GMMHMM] = {}
        self.classes: List[str] = []
        self.is_fitted = False

    def _create_model(self) -> hmm.GMMHMM:
        """Crea un nuevo modelo HMM-GMM."""
        model = hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False
        )
        return model

    def fit(
        self,
        X: List[np.ndarray],
        y: List[str],
        verbose: bool = True
    ):
        """
        Entrena un modelo HMM por cada clase.

        Args:
            X: Lista de matrices de características (cada una: n_frames x n_features)
            y: Lista de etiquetas correspondientes
            verbose: Mostrar progreso
        """
        self.classes = list(set(y))

        # Agrupar datos por clase
        class_data: Dict[str, List[np.ndarray]] = {c: [] for c in self.classes}
        for features, label in zip(X, y):
            # Transponer si es necesario (hmmlearn espera n_samples x n_features)
            if features.shape[0] < features.shape[1]:
                features = features.T
            class_data[label].append(features)

        # Entrenar un modelo por clase
        for i, word in enumerate(self.classes):
            if verbose:
                print(f"  [{i+1}/{len(self.classes)}] Entrenando modelo para '{word}'...")

            # Concatenar todas las secuencias de esta clase
            sequences = class_data[word]
            lengths = [len(seq) for seq in sequences]
            X_concat = np.vstack(sequences)

            # Crear y entrenar modelo
            model = self._create_model()
            try:
                model.fit(X_concat, lengths)
                self.models[word] = model
            except Exception as e:
                print(f"    Error entrenando '{word}': {e}")
                # Crear modelo con menos estados como fallback
                try:
                    model = hmm.GMMHMM(
                        n_components=3,
                        n_mix=4,
                        covariance_type="diag",
                        n_iter=self.n_iter,
                        random_state=self.random_state
                    )
                    model.fit(X_concat, lengths)
                    self.models[word] = model
                except Exception as e2:
                    print(f"    Fallback también falló: {e2}")

        self.is_fitted = True
        if verbose:
            print(f"Entrenamiento completado. {len(self.models)} modelos creados.")

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calcula la probabilidad de cada clase para una secuencia.

        Args:
            X: Matriz de características (n_frames x n_features)

        Returns:
            Diccionario con log-probabilidades por clase
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")

        # Transponer si es necesario
        if X.shape[0] < X.shape[1]:
            X = X.T

        scores = {}
        for word, model in self.models.items():
            try:
                score = model.score(X)
                scores[word] = score
            except Exception:
                scores[word] = float('-inf')

        return scores

    def predict(self, X: np.ndarray) -> str:
        """
        Predice la clase para una secuencia.

        Args:
            X: Matriz de características

        Returns:
            Etiqueta de la clase predicha
        """
        scores = self.predict_proba(X)
        return max(scores, key=scores.get)

    def predict_batch(self, X_list: List[np.ndarray]) -> List[str]:
        """
        Predice clases para múltiples secuencias.

        Args:
            X_list: Lista de matrices de características

        Returns:
            Lista de predicciones
        """
        return [self.predict(X) for X in X_list]

    def score(
        self,
        X: List[np.ndarray],
        y: List[str]
    ) -> float:
        """
        Calcula la precisión del modelo.

        Args:
            X: Lista de matrices de características
            y: Lista de etiquetas verdaderas

        Returns:
            Accuracy
        """
        predictions = self.predict_batch(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)

    def save(self, path: str):
        """Guarda el modelo en disco."""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'classes': self.classes,
                'n_states': self.n_states,
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'is_fitted': self.is_fitted
            }, f)

    def load(self, path: str):
        """Carga el modelo desde disco."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.classes = data['classes']
            self.n_states = data['n_states']
            self.n_components = data['n_components']
            self.covariance_type = data['covariance_type']
            self.is_fitted = data['is_fitted']


class SimpleGMMClassifier:
    """
    Clasificador más simple basado solo en GMM (sin HMM).
    Usa características agregadas en lugar de secuencias.
    """

    def __init__(self, n_components: int = 8, covariance_type: str = "diag"):
        from sklearn.mixture import GaussianMixture
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.models: Dict[str, GaussianMixture] = {}
        self.classes: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena un GMM por clase.

        Args:
            X: Matriz de características agregadas (n_samples x n_features)
            y: Etiquetas
        """
        from sklearn.mixture import GaussianMixture

        self.classes = list(set(y))

        for word in self.classes:
            mask = np.array(y) == word
            X_word = X[mask]

            model = GaussianMixture(
                n_components=min(self.n_components, len(X_word)),
                covariance_type=self.covariance_type,
                random_state=42
            )
            model.fit(X_word)
            self.models[word] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases para múltiples muestras."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        predictions = []
        for x in X:
            scores = {word: model.score(x.reshape(1, -1))
                     for word, model in self.models.items()}
            predictions.append(max(scores, key=scores.get))

        return np.array(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
