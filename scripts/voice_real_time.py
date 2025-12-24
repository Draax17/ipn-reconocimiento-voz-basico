#!/usr/bin/env python3
"""
Script para reconocimiento de voz en tiempo real usando micrófono.
"""
import sys
import time
import queue
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict import VoiceRecognizer

# Configuración de audio
SAMPLE_RATE = 16000  # Frecuencia de muestreo estándar
CHANNELS = 1         # Mono
DURATION = 2.0       # Duración de grabación en segundos (ajustable)

def record_audio(duration: float = DURATION, fs: int = SAMPLE_RATE):
    """
    Graba audio del micrófono.
    """
    print(f"\nGrabando por {duration} segundos... (HABLA AHORA)")
    
    # Grabación síncrona
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=CHANNELS, dtype='float32')
    sd.wait()  # Esperar a que termine la grabación
    
    print("Grabación terminada.")
    return recording.flatten()  # Convertir a 1D array

def save_wav(audio_data, fs, filename="temp_recording.wav"):
    """Guarda el audio en un archivo temporal."""
    # Escalar a 16-bit PCM si es necesario, pero scipy maneja float32 bien para lectura posterior
    # Sin embargo, para compatibilidad con librosa (usado en preprocessing), mejor asegurar formato estándar
    # predict.py usa load_audio de librosa que maneja esto.
    
    # Asegurar que esté en el rango [-1, 1]
    audio_data = np.clip(audio_data, -1, 1)
    
    wav.write(filename, fs, audio_data)
    return filename

def main():
    parser = argparse.ArgumentParser(description='Reconocimiento de Voz en Tiempo Real')
    parser.add_argument('--model', type=str, default='svm',
                       choices=['hmm', 'svm', 'rf', 'mlp', 'all'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duración de la grabación en segundos')
    args = parser.parse_args()

    print("=" * 60)
    print(f"RECONOCIMIENTO EN TIEMPO REAL (Modelo: {args.model.upper()})")
    print("=" * 60)

    # Cargar reconocedor
    print("\nCargando modelo...")
    try:
        if args.model == 'all':
             # Si el usuario elige 'all' en main.py, por defecto usamos SVM para real-time por rapidez/precisión
             model_type = 'svm'
        else:
            model_type = args.model
            
        recognizer = VoiceRecognizer(model_type=model_type)
        recognizer.load_model()
    except Exception as e:
        print(f"\nERROR al cargar el modelo: {e}")
        print("Asegúrate de haber entrenado los modelos primero.")
        return

    print("\nInstrucciones:")
    print(" - Presiona ENTER para empezar a grabar una palabra.")
    print(" - Escribe 'q' y presiona ENTER para salir.")

    temp_file = Path("temp_realtime.wav")

    try:
        while True:
            user_input = input("\n> Presiona ENTER para grabar (o 'q' para salir): ").strip().lower()
            
            if user_input == 'q':
                break
            
            # Grabar
            audio_data = record_audio(duration=args.duration)
            
            # Guardar temporalmente
            save_wav(audio_data, SAMPLE_RATE, str(temp_file))
            
            # Predecir
            print("Analizando...")
            word, confidence = recognizer.predict(str(temp_file))
            
            if word:
                print(f"\nPREDICCIÓN: {word.upper()}")
                print(f"Confianza: {confidence:.2f}")
            else:
                print("\nNo se detectó ninguna palabra clara.")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        # Limpieza
        if temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
        print("\n¡Hasta luego!")

if __name__ == "__main__":
    main()
