"""
Módulo para carga y gestión de datos.
"""
import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, FEATURES_DIR, VOCABULARY,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, MAX_SAMPLES_PER_WORD,
    SAMPLE_RATE
)
from src.preprocessing.audio_converter import load_audio
from src.preprocessing.normalizer import preprocess_audio
from src.preprocessing.vad import trim_silence
from src.features.mfcc_extractor import extract_mfcc_with_deltas, aggregate_features
from src.features.feature_normalizer import FeatureNormalizer


def load_dataset(
    data_dir: Path = PROCESSED_DATA_DIR,
    vocabulary: List[str] = VOCABULARY,
    max_samples_per_word: Optional[int] = MAX_SAMPLES_PER_WORD,
    preprocess: bool = True,
    trim: bool = True,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Carga el dataset de archivos de audio.

    Args:
        data_dir: Directorio con los audios procesados
        vocabulary: Lista de palabras a cargar
        max_samples_per_word: Máximo de muestras por palabra
        preprocess: Aplicar preprocesamiento
        trim: Recortar silencios
        verbose: Mostrar progreso

    Returns:
        Tuple (audios, etiquetas, rutas_archivos)
    """
    audios = []
    labels = []
    file_paths = []

    for word in vocabulary:
        word_dir = data_dir / word
        if not word_dir.exists():
            if verbose:
                print(f"Advertencia: No se encontró carpeta para '{word}'")
            continue

        wav_files = list(word_dir.glob("*.wav"))
        if max_samples_per_word:
            wav_files = wav_files[:max_samples_per_word]

        if verbose:
            print(f"Cargando {len(wav_files)} archivos de '{word}'...")

        for wav_file in wav_files:
            audio, sr = load_audio(str(wav_file))
            if audio is None:
                continue

            if preprocess:
                audio = preprocess_audio(audio)

            if trim:
                audio = trim_silence(audio, sr)

            if len(audio) < sr * 0.1:  # Mínimo 100ms
                continue

            audios.append(audio)
            labels.append(word)
            file_paths.append(str(wav_file))

    if verbose:
        print(f"Dataset cargado: {len(audios)} muestras")

    return audios, labels, file_paths


def extract_features_from_dataset(
    audios: List[np.ndarray],
    labels: List[str],
    aggregate: bool = False,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Extrae características de una lista de audios.

    Args:
        audios: Lista de señales de audio
        labels: Lista de etiquetas
        aggregate: Agregar características temporales en vector fijo
        normalize: Aplicar normalización CMVN
        verbose: Mostrar progreso

    Returns:
        Tuple (características, etiquetas)
    """
    features_list = []
    valid_labels = []

    iterator = tqdm(zip(audios, labels), total=len(audios), desc="Extrayendo features") if verbose else zip(audios, labels)

    for audio, label in iterator:
        try:
            # Extraer MFCCs + deltas
            features = extract_mfcc_with_deltas(audio)

            if features.shape[1] < 5:  # Mínimo 5 frames
                continue

            features_list.append(features)
            valid_labels.append(label)
        except Exception as e:
            if verbose:
                print(f"Error extrayendo features: {e}")
            continue

    # Normalizar
    if normalize and features_list:
        normalizer = FeatureNormalizer(method="cmvn")
        features_list = [normalizer.transform(f) for f in features_list]

    # Agregar si es necesario
    if aggregate:
        features_list = [aggregate_features(f) for f in features_list]

    return features_list, valid_labels


def split_dataset(
    X: List,
    y: List,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    random_state: int = RANDOM_SEED
) -> Dict[str, Tuple]:
    """
    Divide el dataset en train/val/test.

    Args:
        X: Datos
        y: Etiquetas
        train_ratio: Proporción de entrenamiento
        val_ratio: Proporción de validación
        test_ratio: Proporción de prueba
        random_state: Semilla

    Returns:
        Diccionario con 'train', 'val', 'test'
    """
    # Convertir a arrays numpy si es necesario
    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        # Es una lista de arrays (secuencias)
        X_arr = np.arange(len(X))  # Usar índices
    else:
        X_arr = np.array(X)

    y_arr = np.array(y)

    # Primera división: train vs (val + test)
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        X_arr, y_arr,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y_arr
    )

    # Segunda división: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=y_temp
    )

    # Reconstruir listas
    if isinstance(X, list):
        X_train = [X[i] for i in X_train_idx]
        X_val = [X[i] for i in X_val_idx]
        X_test = [X[i] for i in X_test_idx]
    else:
        X_train = X_arr[X_train_idx]
        X_val = X_arr[X_val_idx]
        X_test = X_arr[X_test_idx]

    return {
        'train': (X_train, list(y_train)),
        'val': (X_val, list(y_val)),
        'test': (X_test, list(y_test))
    }


def save_features(
    features: List[np.ndarray],
    labels: List[str],
    path: str
):
    """Guarda características en disco."""
    with open(path, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)


def load_features(path: str) -> Tuple[List[np.ndarray], List[str]]:
    """Carga características desde disco."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data['features'], data['labels']


def get_class_distribution(labels: List[str]) -> Dict[str, int]:
    """Obtiene la distribución de clases."""
    distribution = {}
    for label in labels:
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))


def prepare_data_for_training(
    data_dir: Path = PROCESSED_DATA_DIR,
    vocabulary: List[str] = VOCABULARY,
    max_samples: Optional[int] = MAX_SAMPLES_PER_WORD,
    use_aggregated: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Pipeline completo para preparar datos para entrenamiento.

    Args:
        data_dir: Directorio de datos
        vocabulary: Vocabulario
        max_samples: Máximo de muestras por palabra
        use_aggregated: Usar características agregadas
        save_path: Ruta para guardar características
        verbose: Mostrar progreso

    Returns:
        Diccionario con datos divididos y preparados
    """
    # Cargar audios
    if verbose:
        print("=== Cargando dataset ===")
    audios, labels, paths = load_dataset(
        data_dir, vocabulary, max_samples, verbose=verbose
    )

    # Extraer características
    if verbose:
        print("\n=== Extrayendo características ===")
    features, labels = extract_features_from_dataset(
        audios, labels, aggregate=use_aggregated, verbose=verbose
    )

    # Dividir dataset
    if verbose:
        print("\n=== Dividiendo dataset ===")
    splits = split_dataset(features, labels)

    if verbose:
        print(f"  Train: {len(splits['train'][0])} muestras")
        print(f"  Val: {len(splits['val'][0])} muestras")
        print(f"  Test: {len(splits['test'][0])} muestras")

    # Guardar si se especifica ruta
    if save_path:
        save_features(features, labels, save_path)
        if verbose:
            print(f"\nCaracterísticas guardadas en: {save_path}")

    return {
        'splits': splits,
        'all_features': features,
        'all_labels': labels,
        'class_distribution': get_class_distribution(labels)
    }
