"""
Módulo para conversión de archivos de audio.
Convierte archivos .opus a .wav con los parámetros requeridos.
"""
import os
from pathlib import Path
from typing import List, Optional
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import soundfile as sf

# Configurar la ruta de ffmpeg para Windows (winget install)
FFMPEG_PATH = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin"
FFMPEG_EXE = FFMPEG_PATH / "ffmpeg.exe" if FFMPEG_PATH.exists() else "ffmpeg"

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SAMPLE_RATE, AUDIO_CHANNELS, DATASET_PATH, PROCESSED_DATA_DIR, VOCABULARY


def convert_opus_to_wav(
    input_path: str,
    output_path: str,
    target_sr: int = SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS
) -> bool:
    """
    Convierte un archivo .opus a .wav usando ffmpeg directamente.

    Args:
        input_path: Ruta al archivo .opus
        output_path: Ruta de salida para el archivo .wav
        target_sr: Frecuencia de muestreo objetivo (default: 16000 Hz)
        channels: Número de canales (default: 1, mono)

    Returns:
        True si la conversión fue exitosa, False en caso contrario
    """
    try:
        # Resolver rutas a rutas absolutas
        input_file = Path(input_path).resolve()
        output_file = Path(output_path).resolve()

        # Verificar que el archivo de entrada exista
        if not input_file.exists():
            return False

        # Asegurarse de que el directorio de salida exista
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Si el archivo de salida ya existe, omitir
        if output_file.exists():
            return True

        # Usar ffmpeg directamente (sin especificar formato de entrada)
        cmd = [
            str(FFMPEG_EXE), "-y", "-i", str(input_file),
            "-ar", str(target_sr),
            "-ac", str(channels),
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            str(output_file)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0
    except Exception as e:
        return False


def convert_dataset(
    input_dir: Path = DATASET_PATH,
    output_dir: Path = PROCESSED_DATA_DIR,
    vocabulary: List[str] = VOCABULARY,
    max_samples: Optional[int] = None,
    n_workers: int = 4
) -> dict:
    """
    Convierte todos los archivos .opus del dataset a .wav.

    Args:
        input_dir: Directorio del dataset original
        output_dir: Directorio de salida
        vocabulary: Lista de palabras a procesar
        max_samples: Número máximo de muestras por palabra (None = todas)
        n_workers: Número de workers para procesamiento paralelo

    Returns:
        Diccionario con estadísticas de conversión
    """
    stats = {"total": 0, "success": 0, "failed": 0, "words": {}}

    # Crear lista de tareas
    tasks = []

    for word in vocabulary:
        word_input_dir = input_dir / word
        word_output_dir = output_dir / word

        if not word_input_dir.exists():
            print(f"Advertencia: No se encontró carpeta para '{word}'")
            continue

        word_output_dir.mkdir(parents=True, exist_ok=True)

        # Obtener archivos .opus y verificar que existan
        opus_files = [f for f in word_input_dir.glob("*.opus") if f.exists() and f.is_file()]

        if max_samples:
            opus_files = opus_files[:max_samples]

        for opus_file in opus_files:
            wav_file = word_output_dir / opus_file.name.replace(".opus", ".wav")
            # Solo agregar si el archivo de entrada realmente existe
            if opus_file.exists():
                tasks.append((str(opus_file.resolve()), str(wav_file.resolve()), word))

    stats["total"] = len(tasks)
    print(f"Convirtiendo {len(tasks)} archivos de audio...")

    # Procesar en paralelo
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(convert_opus_to_wav, inp, out): (inp, out, word)
            for inp, out, word in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Convirtiendo"):
            inp, out, word = futures[future]
            try:
                success = future.result()
                if success:
                    stats["success"] += 1
                    stats["words"][word] = stats["words"].get(word, 0) + 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                stats["failed"] += 1
                print(f"Error: {e}")

    print(f"\nConversión completada: {stats['success']}/{stats['total']} exitosos")
    return stats


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> tuple:
    """
    Carga un archivo de audio y retorna la señal y frecuencia de muestreo.

    Args:
        file_path: Ruta al archivo de audio
        sr: Frecuencia de muestreo deseada

    Returns:
        Tuple (señal, frecuencia_de_muestreo)
    """
    try:
        # Intentar con soundfile primero
        audio, orig_sr = sf.read(file_path)

        # Si es estéreo, convertir a mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resamplear si es necesario
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        return audio.astype(np.float32), sr
    except Exception as e:
        print(f"Error cargando {file_path}: {e}")
        return None, None


if __name__ == "__main__":
    # Test de conversión
    print("Iniciando conversión del dataset...")
    stats = convert_dataset(max_samples=10)  # Solo 10 por palabra para prueba
    print(f"Estadísticas: {stats}")
