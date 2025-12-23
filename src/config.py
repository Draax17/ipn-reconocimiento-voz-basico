"""
Configuración global del sistema de reconocimiento de voz.
"""
import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset original
DATASET_PATH = PROJECT_ROOT / "dataset_filtrado" / "dataset_filtrado"

# Parámetros de audio
SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1   # Mono
BIT_DEPTH = 16       # bits

# Parámetros de preprocesamiento
PRE_EMPHASIS_COEF = 0.97
VAD_ENERGY_THRESHOLD = -40  # dB
VAD_MIN_DURATION = 0.04     # segundos
NOISE_REDUCTION_ENABLED = True

# Parámetros de MFCCs
N_MFCC = 13
N_FFT = 512              # ~32ms a 16kHz
HOP_LENGTH = 160         # ~10ms a 16kHz
N_MELS = 40
FMIN = 80                # Hz
FMAX = 8000              # Hz
INCLUDE_DELTA = True
INCLUDE_DELTA_DELTA = True

# Vocabulario seleccionado (18 palabras)
VOCABULARY = [
    "agua",
    "con",
    "del",
    "dos",
    "entre",
    "esta",
    "fue",
    "gran",
    "las",
    "los",
    "más",
    "muy",
    "para",
    "por",
    "que",
    "son",
    "una",
    "vida"
]

# Parámetros de entrenamiento
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
MAX_SAMPLES_PER_WORD = 300  # Limitar para no sobrecargar memoria

# Parámetros del modelo HMM
HMM_N_STATES = 5
HMM_N_COMPONENTS = 8
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 100

# Parámetros del modelo RNN (opcional)
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.3
RNN_LEARNING_RATE = 0.001
RNN_BATCH_SIZE = 32
RNN_EPOCHS = 50

# Crear directorios si no existen
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
