"""
Módulo para visualizaciones de resultados.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Matriz de Confusión",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "Blues",
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """
    Grafica la matriz de confusión.

    Args:
        cm: Matriz de confusión
        labels: Lista de etiquetas
        title: Título del gráfico
        figsize: Tamaño de la figura
        cmap: Mapa de colores
        normalize: Normalizar por fila
        save_path: Ruta para guardar
    """
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Curvas de Entrenamiento",
    save_path: Optional[str] = None
):
    """
    Grafica las curvas de entrenamiento.

    Args:
        train_losses: Pérdidas de entrenamiento
        val_losses: Pérdidas de validación
        train_accs: Accuracies de entrenamiento
        val_accs: Accuracies de validación
        title: Título
        save_path: Ruta para guardar
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Gráfico de pérdida
    axes[0].plot(epochs, train_losses, 'b-', label='Train')
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='Validation')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].set_title('Pérdida por Época')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gráfico de accuracy
    if train_accs:
        axes[1].plot(epochs, train_accs, 'b-', label='Train')
        if val_accs:
            axes[1].plot(epochs, val_accs, 'r-', label='Validation')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy por Época')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No hay datos de accuracy',
                    ha='center', va='center', transform=axes[1].transAxes)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curvas guardadas en: {save_path}")

    plt.show()


def plot_class_distribution(
    distribution: Dict[str, int],
    title: str = "Distribución de Clases",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Grafica la distribución de clases.

    Args:
        distribution: Diccionario {clase: conteo}
        title: Título
        figsize: Tamaño de figura
        save_path: Ruta para guardar
    """
    classes = list(distribution.keys())
    counts = list(distribution.values())

    plt.figure(figsize=figsize)
    bars = plt.bar(classes, counts, color='steelblue', edgecolor='navy')

    # Añadir valores sobre las barras
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Número de Muestras', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribución guardada en: {save_path}")

    plt.show()


def plot_per_class_accuracy(
    per_class: Dict[str, float],
    title: str = "Accuracy por Clase",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Grafica el accuracy por clase.

    Args:
        per_class: Diccionario {clase: accuracy}
        title: Título
        figsize: Tamaño de figura
        save_path: Ruta para guardar
    """
    # Ordenar por accuracy
    sorted_items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0] for x in sorted_items]
    accs = [x[1] for x in sorted_items]

    plt.figure(figsize=figsize)

    colors = ['green' if a >= 0.8 else 'orange' if a >= 0.6 else 'red' for a in accs]
    bars = plt.bar(classes, accs, color=colors, edgecolor='black')

    # Línea de referencia
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Objetivo (80%)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Mínimo (60%)')

    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Accuracy por clase guardado en: {save_path}")

    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'mean',
    title: str = "Comparación de Modelos",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compara el rendimiento de múltiples modelos.

    Args:
        results: Diccionario {modelo: {metric: valor}}
        metric: Métrica a graficar
        title: Título
        figsize: Tamaño de figura
        save_path: Ruta para guardar
    """
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    if 'std' in results[models[0]]:
        errors = [results[m]['std'] for m in models]
    else:
        errors = None

    plt.figure(figsize=figsize)

    bars = plt.bar(models, values, yerr=errors, capsize=5,
                  color='steelblue', edgecolor='navy')

    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    # Añadir valores
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparación guardada en: {save_path}")

    plt.show()


def plot_mfcc_features(
    mfcc: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    title: str = "Características MFCC",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Visualiza las características MFCC.

    Args:
        mfcc: Matriz MFCC (n_mfcc, n_frames)
        sr: Frecuencia de muestreo
        hop_length: Salto entre frames
        title: Título
        figsize: Tamaño de figura
        save_path: Ruta para guardar
    """
    plt.figure(figsize=figsize)

    # Calcular eje temporal
    n_frames = mfcc.shape[1]
    times = np.arange(n_frames) * hop_length / sr

    plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis',
               extent=[times[0], times[-1], 0, mfcc.shape[0]])

    plt.colorbar(label='Amplitud')
    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Coeficiente MFCC', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"MFCCs guardados en: {save_path}")

    plt.show()


def plot_waveform_and_features(
    audio: np.ndarray,
    mfcc: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    title: str = "Audio y Características",
    save_path: Optional[str] = None
):
    """
    Visualiza la forma de onda junto con las características MFCC.

    Args:
        audio: Señal de audio
        mfcc: Matriz MFCC
        sr: Frecuencia de muestreo
        hop_length: Salto entre frames
        title: Título
        save_path: Ruta para guardar
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Forma de onda
    times_audio = np.arange(len(audio)) / sr
    axes[0].plot(times_audio, audio, 'b', linewidth=0.5)
    axes[0].set_xlabel('Tiempo (s)')
    axes[0].set_ylabel('Amplitud')
    axes[0].set_title('Forma de Onda')
    axes[0].grid(True, alpha=0.3)

    # MFCCs
    n_frames = mfcc.shape[1]
    times_mfcc = np.arange(n_frames) * hop_length / sr
    im = axes[1].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis',
                        extent=[times_mfcc[0], times_mfcc[-1], 0, mfcc.shape[0]])
    fig.colorbar(im, ax=axes[1], label='Amplitud')
    axes[1].set_xlabel('Tiempo (s)')
    axes[1].set_ylabel('Coeficiente MFCC')
    axes[1].set_title('Características MFCC')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")

    plt.show()
