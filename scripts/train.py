#!/usr/bin/env python3
"""
Script para entrenar los modelos de reconocimiento de voz.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, VOCABULARY,
    MAX_SAMPLES_PER_WORD, FEATURES_DIR
)
from src.utils.data_loader import (
    load_dataset, extract_features_from_dataset, split_dataset,
    save_features, get_class_distribution
)
from src.models.hmm_model import HMMClassifier
from src.models.classifier import (
    train_svm, train_random_forest, train_mlp, evaluate_classifiers, save_model
)
from src.features.mfcc_extractor import aggregate_features
from src.evaluation.metrics import calculate_metrics, generate_evaluation_report
from src.evaluation.visualizations import (
    plot_confusion_matrix, plot_class_distribution, plot_model_comparison
)


def train_hmm_model(X_train, y_train, X_val, y_val, verbose=True):
    """Entrena y evalúa modelo HMM."""
    if verbose:
        print("\n" + "=" * 50)
        print("ENTRENANDO MODELO HMM-GMM")
        print("=" * 50)

    model = HMMClassifier()
    model.fit(X_train, y_train, verbose=verbose)

    # Evaluar en validación
    if verbose:
        print("\nEvaluando en conjunto de validación...")

    y_pred = model.predict_batch(X_val)
    metrics = calculate_metrics(y_val, y_pred)

    if verbose:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")

    return model, metrics


def train_classic_models(X_train, y_train, X_val, y_val, verbose=True):
    """Entrena y evalúa modelos clásicos (SVM, RF, etc.)."""
    if verbose:
        print("\n" + "=" * 50)
        print("ENTRENANDO MODELOS CLÁSICOS")
        print("=" * 50)

    # Agregar features para modelos clásicos
    X_train_agg = np.array([aggregate_features(x) for x in X_train])
    X_val_agg = np.array([aggregate_features(x) for x in X_val])

    models = {}
    results = {}

    # SVM
    if verbose:
        print("\nEntrenando SVM...")
    svm = train_svm(X_train_agg, y_train)
    svm_pred = svm.predict(X_val_agg)
    svm_metrics = calculate_metrics(y_val, svm_pred)
    models['svm'] = svm
    results['SVM'] = svm_metrics
    if verbose:
        print(f"  Accuracy: {svm_metrics['accuracy']:.4f}")

    # Random Forest
    if verbose:
        print("\nEntrenando Random Forest...")
    rf = train_random_forest(X_train_agg, y_train)
    rf_pred = rf.predict(X_val_agg)
    rf_metrics = calculate_metrics(y_val, rf_pred)
    models['rf'] = rf
    results['Random Forest'] = rf_metrics
    if verbose:
        print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")

    # MLP
    if verbose:
        print("\nEntrenando MLP...")
    mlp = train_mlp(X_train_agg, y_train)
    mlp_pred = mlp.predict(X_val_agg)
    mlp_metrics = calculate_metrics(y_val, mlp_pred)
    models['mlp'] = mlp
    results['MLP'] = mlp_metrics
    if verbose:
        print(f"  Accuracy: {mlp_metrics['accuracy']:.4f}")

    return models, results


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos de reconocimiento de voz')
    parser.add_argument('--model', type=str, default='all',
                       choices=['hmm', 'svm', 'rf', 'mlp', 'all'],
                       help='Modelo a entrenar')
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES_PER_WORD,
                       help='Máximo de muestras por palabra')
    parser.add_argument('--no-plots', action='store_true',
                       help='No mostrar gráficas')
    args = parser.parse_args()

    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS")
    print("=" * 60)

    # Cargar datos
    print("\n[1/4] Cargando dataset...")
    audios, labels, paths = load_dataset(
        data_dir=PROCESSED_DATA_DIR,
        vocabulary=VOCABULARY,
        max_samples_per_word=args.max_samples,
        verbose=True
    )

    if len(audios) == 0:
        print("\nERROR: No se encontraron datos.")
        print("Ejecuta primero: python scripts/prepare_data.py")
        sys.exit(1)

    # Extraer características
    print("\n[2/4] Extrayendo características...")
    features, labels = extract_features_from_dataset(audios, labels, verbose=True)

    # Mostrar distribución
    distribution = get_class_distribution(labels)
    print(f"\nDistribución de clases:")
    for word, count in list(distribution.items())[:5]:
        print(f"  {word}: {count}")
    if len(distribution) > 5:
        print(f"  ... y {len(distribution) - 5} más")

    if not args.no_plots:
        plot_class_distribution(
            distribution,
            title="Distribución del Dataset",
            save_path=str(RESULTS_DIR / "distribucion_clases.png")
        )

    # Dividir datos
    print("\n[3/4] Dividiendo dataset...")
    splits = split_dataset(features, labels)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"  Train: {len(X_train)} muestras")
    print(f"  Val: {len(X_val)} muestras")
    print(f"  Test: {len(X_test)} muestras")

    # Guardar características
    features_path = FEATURES_DIR / "features.pkl"
    save_features(features, labels, str(features_path))
    print(f"\nCaracterísticas guardadas en: {features_path}")

    # Entrenar modelos
    print("\n[4/4] Entrenando modelos...")
    all_results = {}

    if args.model in ['hmm', 'all']:
        hmm_model, hmm_metrics = train_hmm_model(X_train, y_train, X_val, y_val)
        hmm_model.save(str(MODELS_DIR / "hmm_model.pkl"))
        all_results['HMM-GMM'] = hmm_metrics
        print(f"Modelo HMM guardado en: {MODELS_DIR / 'hmm_model.pkl'}")

    if args.model in ['svm', 'rf', 'mlp', 'all']:
        classic_models, classic_results = train_classic_models(
            X_train, y_train, X_val, y_val
        )
        all_results.update(classic_results)

        # Guardar modelos
        for name, model in classic_models.items():
            model_path = MODELS_DIR / f"{name}_model.pkl"
            save_model(model, str(model_path))
            print(f"Modelo {name.upper()} guardado en: {model_path}")

    # Resumen de resultados
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS (Validación)")
    print("=" * 60)
    print(f"{'Modelo':<20} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 44)
    for name, metrics in all_results.items():
        print(f"{name:<20} {metrics['accuracy']:.4f}       {metrics['f1']:.4f}")

    # Encontrar mejor modelo
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nMejor modelo: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

    # Comparación visual
    if not args.no_plots and len(all_results) > 1:
        comparison_data = {name: {'mean': m['accuracy'], 'std': 0}
                         for name, m in all_results.items()}
        plot_model_comparison(
            comparison_data,
            title="Comparación de Modelos",
            save_path=str(RESULTS_DIR / "comparacion_modelos.png")
        )

    print("\n¡Entrenamiento completado!")
    print(f"Modelos guardados en: {MODELS_DIR}")
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
