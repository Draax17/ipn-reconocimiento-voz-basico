"""
Módulo para clasificadores tradicionales de ML.
Incluye SVM, Random Forest y otros clasificadores.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    probability: bool = True
) -> Pipeline:
    """
    Entrena un clasificador SVM con escalado de características.

    Args:
        X_train: Datos de entrenamiento (n_samples, n_features)
        y_train: Etiquetas
        kernel: Tipo de kernel ("rbf", "linear", "poly")
        C: Parámetro de regularización
        gamma: Parámetro gamma del kernel
        probability: Habilitar estimación de probabilidades

    Returns:
        Pipeline entrenado con scaler + SVM
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2
) -> Pipeline:
    """
    Entrena un clasificador Random Forest.

    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
        min_samples_split: Mínimo de muestras para dividir

    Returns:
        Pipeline entrenado
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3
) -> Pipeline:
    """
    Entrena un clasificador Gradient Boosting.

    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas
        n_estimators: Número de estimadores
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima

    Returns:
        Pipeline entrenado
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    activation: str = "relu",
    max_iter: int = 500
) -> Pipeline:
    """
    Entrena un perceptrón multicapa.

    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas
        hidden_layer_sizes: Tamaño de capas ocultas
        activation: Función de activación
        max_iter: Máximo de iteraciones

    Returns:
        Pipeline entrenado
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


def optimize_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Optimiza hiperparámetros de SVM usando grid search.

    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas
        cv: Número de folds para validación cruzada

    Returns:
        Tuple (mejor_modelo, mejores_parámetros)
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, float]:
    """
    Evalúa múltiples clasificadores usando validación cruzada.

    Args:
        X: Datos
        y: Etiquetas
        cv: Número de folds

    Returns:
        Diccionario con accuracy promedio por clasificador
    """
    classifiers = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42))
        ]),
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))
        ])
    }

    results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results


def save_model(model: Pipeline, path: str):
    """Guarda un modelo en disco."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> Pipeline:
    """Carga un modelo desde disco."""
    with open(path, 'rb') as f:
        return pickle.load(f)
