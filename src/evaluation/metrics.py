"""
Módulo para cálculo de métricas de evaluación.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    average: str = "macro"
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        average: Tipo de promedio ("macro", "micro", "weighted")

    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics


def calculate_wer(
    y_true: List[str],
    y_pred: List[str]
) -> float:
    """
    Calcula Word Error Rate.
    Para palabras aisladas, WER = 1 - Accuracy.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas

    Returns:
        WER como porcentaje
    """
    errors = sum(t != p for t, p in zip(y_true, y_pred))
    return errors / len(y_true)


def calculate_top_k_accuracy(
    y_true: List[str],
    y_pred_proba: List[Dict[str, float]],
    k: int = 3
) -> float:
    """
    Calcula top-k accuracy.

    Args:
        y_true: Etiquetas verdaderas
        y_pred_proba: Lista de diccionarios con probabilidades por clase
        k: Número de predicciones top a considerar

    Returns:
        Top-k accuracy
    """
    correct = 0
    for true_label, proba in zip(y_true, y_pred_proba):
        # Obtener top-k predicciones
        top_k = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:k]
        top_k_labels = [label for label, _ in top_k]
        if true_label in top_k_labels:
            correct += 1

    return correct / len(y_true)


def get_classification_report(
    y_true: List[str],
    y_pred: List[str],
    output_dict: bool = False
) -> str:
    """
    Genera reporte de clasificación detallado.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        output_dict: Retornar como diccionario

    Returns:
        Reporte como string o diccionario
    """
    return classification_report(y_true, y_pred, zero_division=0, output_dict=output_dict)


def get_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Calcula la matriz de confusión.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        labels: Lista de etiquetas en orden

    Returns:
        Tuple (matriz_confusion, lista_etiquetas)
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def get_most_confused_pairs(
    y_true: List[str],
    y_pred: List[str],
    top_n: int = 10
) -> List[Tuple[str, str, int]]:
    """
    Encuentra los pares de clases más confundidos.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        top_n: Número de pares a retornar

    Returns:
        Lista de tuplas (clase_real, clase_predicha, conteo)
    """
    cm, labels = get_confusion_matrix(y_true, y_pred)

    # Encontrar confusiones (excluyendo diagonal)
    confusions = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append((true_label, pred_label, cm[i, j]))

    # Ordenar por conteo
    confusions.sort(key=lambda x: x[2], reverse=True)

    return confusions[:top_n]


def analyze_errors_by_class(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Analiza errores por clase.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas

    Returns:
        Diccionario con métricas por clase
    """
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    # Filtrar solo las clases (no los promedios)
    class_metrics = {}
    for key, value in report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            class_metrics[key] = value

    return class_metrics


def calculate_per_class_accuracy(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, float]:
    """
    Calcula accuracy por clase.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas

    Returns:
        Diccionario con accuracy por clase
    """
    classes = sorted(list(set(y_true)))
    per_class = {}

    for cls in classes:
        mask = np.array(y_true) == cls
        if mask.sum() > 0:
            correct = np.array(y_pred)[mask] == cls
            per_class[cls] = correct.sum() / mask.sum()
        else:
            per_class[cls] = 0.0

    return per_class


def generate_evaluation_report(
    y_true: List[str],
    y_pred: List[str],
    model_name: str = "Modelo"
) -> str:
    """
    Genera un reporte completo de evaluación.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        model_name: Nombre del modelo

    Returns:
        Reporte como string
    """
    metrics = calculate_metrics(y_true, y_pred)
    wer = calculate_wer(y_true, y_pred)
    confused = get_most_confused_pairs(y_true, y_pred, top_n=5)

    report = []
    report.append(f"=" * 60)
    report.append(f"REPORTE DE EVALUACIÓN: {model_name}")
    report.append(f"=" * 60)
    report.append("")
    report.append("MÉTRICAS GLOBALES:")
    report.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"  Precision: {metrics['precision']:.4f}")
    report.append(f"  Recall:    {metrics['recall']:.4f}")
    report.append(f"  F1-Score:  {metrics['f1']:.4f}")
    report.append(f"  WER:       {wer:.4f} ({wer*100:.2f}%)")
    report.append("")
    report.append("PARES MÁS CONFUNDIDOS:")
    for true_cls, pred_cls, count in confused:
        report.append(f"  '{true_cls}' → '{pred_cls}': {count} errores")
    report.append("")
    report.append("REPORTE POR CLASE:")
    report.append(get_classification_report(y_true, y_pred))

    return "\n".join(report)
