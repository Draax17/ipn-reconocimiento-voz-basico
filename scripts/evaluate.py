#!/usr/bin/env python3
"""
Script para evaluar los modelos entrenados.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, VOCABULARY
from src.utils.data_loader import load_dataset, extract_features_from_dataset, split_dataset
from src.models.hmm_model import HMMClassifier
from src.models.classifier import load_model
from src.features.mfcc_extractor import aggregate_features
from src.evaluation.metrics import (
    calculate_metrics, generate_evaluation_report, get_confusion_matrix,
    get_most_confused_pairs, calculate_per_class_accuracy
)
from src.evaluation.visualizations import (
    plot_confusion_matrix, plot_per_class_accuracy
)


def evaluate_model(model, X_test, y_test, model_name, is_hmm=False, save_dir=None):
    """Evalúa un modelo y genera reportes."""
    print(f"\nEvaluando {model_name}...")

    # Predecir
    if is_hmm:
        y_pred = model.predict_batch(X_test)
    else:
        X_test_agg = np.array([aggregate_features(x) for x in X_test])
        y_pred = model.predict(X_test_agg)

    # Métricas
    metrics = calculate_metrics(y_test, y_pred)

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    # Reporte completo
    report = generate_evaluation_report(y_test, y_pred, model_name)

    if save_dir:
        # Guardar reporte
        report_path = save_dir / f"evaluacion_{model_name.lower().replace(' ', '_')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Reporte guardado en: {report_path}")

        # Matriz de confusión
        cm, labels = get_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, labels,
            title=f"Matriz de Confusión - {model_name}",
            save_path=str(save_dir / f"confusion_{model_name.lower().replace(' ', '_')}.png")
        )

        # Accuracy por clase
        per_class = calculate_per_class_accuracy(y_test, y_pred)
        plot_per_class_accuracy(
            per_class,
            title=f"Accuracy por Clase - {model_name}",
            save_path=str(save_dir / f"per_class_{model_name.lower().replace(' ', '_')}.png")
        )

    return metrics, y_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelos de reconocimiento de voz')
    parser.add_argument('--model', type=str, default='all',
                       choices=['hmm', 'svm', 'rf', 'mlp', 'all'],
                       help='Modelo a evaluar')
    parser.add_argument('--no-plots', action='store_true',
                       help='No mostrar gráficas')
    args = parser.parse_args()

    print("=" * 60)
    print("EVALUACIÓN DE MODELOS")
    print("=" * 60)

    # Cargar datos de prueba
    print("\n[1/3] Cargando dataset...")
    audios, labels, paths = load_dataset(
        data_dir=PROCESSED_DATA_DIR,
        vocabulary=VOCABULARY,
        verbose=True
    )

    if len(audios) == 0:
        print("\nERROR: No se encontraron datos.")
        sys.exit(1)

    # Extraer características
    print("\n[2/3] Extrayendo características...")
    features, labels = extract_features_from_dataset(audios, labels, verbose=True)

    # Dividir datos (usamos la misma semilla para consistencia)
    splits = split_dataset(features, labels)
    X_test, y_test = splits['test']
    print(f"\nConjunto de prueba: {len(X_test)} muestras")

    # Evaluar modelos
    print("\n[3/3] Evaluando modelos...")
    results = {}
    save_dir = RESULTS_DIR if not args.no_plots else None

    # HMM
    if args.model in ['hmm', 'all']:
        hmm_path = MODELS_DIR / "hmm_model.pkl"
        if hmm_path.exists():
            hmm = HMMClassifier()
            hmm.load(str(hmm_path))
            metrics, _ = evaluate_model(
                hmm, X_test, y_test, "HMM-GMM",
                is_hmm=True, save_dir=save_dir
            )
            results['HMM-GMM'] = metrics
        else:
            print(f"\nAdvertencia: No se encontró modelo HMM en {hmm_path}")

    # Modelos clásicos
    classic_models = {'svm': 'SVM', 'rf': 'Random Forest', 'mlp': 'MLP'}

    for model_key, model_name in classic_models.items():
        if args.model in [model_key, 'all']:
            model_path = MODELS_DIR / f"{model_key}_model.pkl"
            if model_path.exists():
                model = load_model(str(model_path))
                metrics, _ = evaluate_model(
                    model, X_test, y_test, model_name,
                    is_hmm=False, save_dir=save_dir
                )
                results[model_name] = metrics
            else:
                print(f"\nAdvertencia: No se encontró modelo {model_name} en {model_path}")

    # Resumen final
    if results:
        print("\n" + "=" * 60)
        print("RESUMEN FINAL (Conjunto de Prueba)")
        print("=" * 60)
        print(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 68)

        best_acc = 0
        best_model = ""
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['accuracy']:.4f}       {metrics['precision']:.4f}       "
                  f"{metrics['recall']:.4f}       {metrics['f1']:.4f}")
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_model = name

        print(f"\nMejor modelo: {best_model} (Accuracy: {best_acc:.4f})")

    print("\n¡Evaluación completada!")
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
