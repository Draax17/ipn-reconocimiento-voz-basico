#!/usr/bin/env python3
"""
Script para preparar los datos del dataset.
Convierte archivos .opus a .wav y los organiza.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATASET_PATH, PROCESSED_DATA_DIR, VOCABULARY, MAX_SAMPLES_PER_WORD
from src.preprocessing.audio_converter import convert_dataset


def main():
    print("=" * 60)
    print("PREPARACIÓN DEL DATASET")
    print("=" * 60)
    print(f"\nDataset origen: {DATASET_PATH}")
    print(f"Destino: {PROCESSED_DATA_DIR}")
    print(f"Vocabulario: {len(VOCABULARY)} palabras")
    print(f"Máximo de muestras por palabra: {MAX_SAMPLES_PER_WORD}")
    print()

    # Verificar que existe el dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: No se encontró el dataset en {DATASET_PATH}")
        print("Asegúrate de que el dataset_filtrado esté correctamente ubicado.")
        sys.exit(1)

    # Mostrar palabras disponibles
    available_words = [d.name for d in DATASET_PATH.iterdir() if d.is_dir()]
    print(f"Palabras disponibles en el dataset: {len(available_words)}")

    # Verificar cuáles del vocabulario están disponibles
    missing = [w for w in VOCABULARY if w not in available_words]
    if missing:
        print(f"\nAdvertencia: Las siguientes palabras no están en el dataset:")
        for w in missing:
            print(f"  - {w}")

    # Confirmar
    input("\nPresiona Enter para comenzar la conversión...")

    # Convertir
    print("\nIniciando conversión...")
    stats = convert_dataset(
        input_dir=DATASET_PATH,
        output_dir=PROCESSED_DATA_DIR,
        vocabulary=VOCABULARY,
        max_samples=MAX_SAMPLES_PER_WORD,
        n_workers=4
    )

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE CONVERSIÓN")
    print("=" * 60)
    print(f"Total de archivos procesados: {stats['total']}")
    print(f"Conversiones exitosas: {stats['success']}")
    print(f"Conversiones fallidas: {stats['failed']}")
    print("\nArchivos por palabra:")
    for word, count in sorted(stats['words'].items()):
        print(f"  {word}: {count}")

    print("\n¡Preparación completada!")
    print(f"Los archivos procesados están en: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
