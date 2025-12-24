#!/usr/bin/env python3
"""
Sistema de Reconocimiento de Voz - ESCOM IPN
Proyecto Final de Reconocimiento de Voz

Este es el punto de entrada principal del sistema.
Permite ejecutar todas las etapas del pipeline de reconocimiento.
"""
import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATASET_PATH, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR,
    VOCABULARY, MAX_SAMPLES_PER_WORD, DATA_DIR
)


def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    required = ['numpy', 'scipy', 'librosa', 'sklearn', 'hmmlearn', 'matplotlib']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("ERROR: Faltan las siguientes dependencias:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstálalas con: pip install -r requirements.txt")
        return False

    return True


def run_prepare():
    """Ejecuta la preparación de datos."""
    from scripts.prepare_data import main as prepare_main
    prepare_main()


def run_train(model='all', max_samples=None):
    """Ejecuta el entrenamiento."""
    from scripts.train import main as train_main
    sys.argv = ['train.py', '--model', model]
    if max_samples:
        sys.argv.extend(['--max-samples', str(max_samples)])
    train_main()


def run_evaluate(model='all'):
    """Ejecuta la evaluación."""
    from scripts.evaluate import main as evaluate_main
    sys.argv = ['evaluate.py', '--model', model]
    evaluate_main()


def run_predict(audio_path, model='svm'):
    """Ejecuta predicción en un audio."""
    from scripts.predict import main as predict_main
    sys.argv = ['predict.py', audio_path, '--model', model]
    predict_main()


def run_real_time(model='svm'):
    """Ejecuta el reconocimiento en tiempo real."""
    from scripts.voice_real_time import main as realtime_main
    sys.argv = ['voice_real_time.py', '--model', model]
    realtime_main()


def run_full_pipeline():
    """Ejecuta el pipeline completo."""
    print("=" * 70)
    print("SISTEMA DE RECONOCIMIENTO DE VOZ - PIPELINE COMPLETO")
    print("ESCOM - Instituto Politécnico Nacional")
    print("=" * 70)

    # Verificar dependencias
    print("\n[0/4] Verificando dependencias...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ Todas las dependencias instaladas")

    # Verificar dataset
    print("\n[1/4] Verificando dataset...")
    if not DATASET_PATH.exists():
        print(f"ERROR: No se encontró el dataset en: {DATASET_PATH}")
        print("Asegúrate de que el dataset esté correctamente ubicado.")
        sys.exit(1)

    n_words = len([d for d in DATASET_PATH.iterdir() if d.is_dir()])
    print(f"✓ Dataset encontrado: {n_words} palabras disponibles")

    # Verificar si ya hay datos procesados
    if PROCESSED_DATA_DIR.exists() and any(PROCESSED_DATA_DIR.iterdir()):
        print(f"✓ Datos procesados encontrados en: {PROCESSED_DATA_DIR}")
        response = input("\n¿Deseas volver a procesar los datos? (s/N): ").strip().lower()
        if response == 's':
            run_prepare()
    else:
        print("\nNo se encontraron datos procesados. Iniciando preparación...")
        run_prepare()

    # Entrenar
    print("\n[2/4] Entrenamiento de modelos...")
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.pkl")):
        response = input("\n¿Deseas volver a entrenar los modelos? (s/N): ").strip().lower()
        if response == 's':
            run_train()
    else:
        run_train()

    # Evaluar
    print("\n[3/4] Evaluación de modelos...")
    run_evaluate()

    # Demo
    print("\n[4/4] Demo de predicción...")
    test_files = list(PROCESSED_DATA_DIR.rglob("*.wav"))[:3]
    if test_files:
        print("\nProbando predicción con algunos archivos:")
        for f in test_files:
            print(f"\n→ {f.name}")
            from scripts.predict import VoiceRecognizer
            recognizer = VoiceRecognizer(model_type='svm')
            recognizer.load_model()
            word, conf = recognizer.predict(str(f))
            print(f"  Predicción: {word}")

    print("\n" + "=" * 70)
    print("¡PIPELINE COMPLETADO!")
    print("=" * 70)
    print(f"\nResultados guardados en: {RESULTS_DIR}")
    print(f"Modelos guardados en: {MODELS_DIR}")


def interactive_menu():
    """Menú interactivo."""
    while True:
        print("\n" + "=" * 50)
        print("SISTEMA DE RECONOCIMIENTO DE VOZ")
        print("=" * 50)
        print("\n1. Ejecutar pipeline completo")
        print("2. Preparar datos (convertir audios)")
        print("3. Entrenar modelos")
        print("4. Evaluar modelos")
        print("5. Evaluar audios propios (data/evaluar)")
        print("6. Reconocimiento en tiempo real")
        print("7. Ver configuración")
        print("0. Salir")

        choice = input("\nSelecciona una opción: ").strip()

        if choice == '1':
            run_full_pipeline()
        elif choice == '2':
            run_prepare()
        elif choice == '3':
            model = input("Modelo (hmm/svm/rf/mlp/all) [all]: ").strip() or 'all'
            run_train(model)
        elif choice == '4':
            model = input("Modelo (hmm/svm/rf/mlp/all) [all]: ").strip() or 'all'
            run_evaluate(model)
        elif choice == '5':
            eval_folder = DATA_DIR / "evaluar"
            if not eval_folder.exists():
                print(f"\nNo existe la carpeta: {eval_folder}")
                print("Crea la carpeta y agrega archivos .wav con el nombre de la palabra esperada.")
            else:
                model = input("Modelo (svm/rf/mlp) [svm]: ").strip() or 'svm'
                run_predict(str(eval_folder), model)
        elif choice == '6':
            model = input("Modelo a utilizar (svm/rf/mlp) [svm]: ").strip() or 'svm'
            run_real_time(model)
        elif choice == '7':
            print("\n--- CONFIGURACIÓN ---")
            print(f"Dataset: {DATASET_PATH}")
            print(f"Datos procesados: {PROCESSED_DATA_DIR}")
            print(f"Modelos: {MODELS_DIR}")
            print(f"Resultados: {RESULTS_DIR}")
            print(f"Vocabulario: {len(VOCABULARY)} palabras")
            print(f"Max muestras por palabra: {MAX_SAMPLES_PER_WORD}")
        elif choice == '0':
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción no válida.")


def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Reconocimiento de Voz - ESCOM IPN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                    # Menú interactivo
  python main.py --full             # Pipeline completo
  python main.py --prepare          # Solo preparar datos
  python main.py --train            # Solo entrenar
  python main.py --evaluate         # Solo evaluar
  python main.py --predict audio.wav  # Predecir un audio
        """
    )

    parser.add_argument('--full', action='store_true',
                       help='Ejecutar pipeline completo')
    parser.add_argument('--prepare', action='store_true',
                       help='Preparar datos (convertir audios)')
    parser.add_argument('--train', action='store_true',
                       help='Entrenar modelos')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluar modelos')
    parser.add_argument('--predict', type=str, metavar='AUDIO',
                       help='Predecir palabra en archivo de audio')
    parser.add_argument('--model', type=str, default='all',
                       choices=['hmm', 'svm', 'rf', 'mlp', 'all'],
                       help='Modelo a usar (default: all)')

    args = parser.parse_args()

    # Si no hay argumentos, mostrar menú interactivo
    if not any([args.full, args.prepare, args.train, args.evaluate, args.predict]):
        interactive_menu()
        return

    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)

    # Ejecutar acción solicitada
    if args.full:
        run_full_pipeline()
    elif args.prepare:
        run_prepare()
    elif args.train:
        run_train(args.model)
    elif args.evaluate:
        run_evaluate(args.model)
    elif args.predict:
        run_predict(args.predict, args.model if args.model != 'all' else 'svm')


if __name__ == "__main__":
    main()
