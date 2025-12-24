#!/usr/bin/env python3
"""
Script para hacer predicciones con los modelos entrenados.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, SAMPLE_RATE
from src.preprocessing.audio_converter import load_audio
from src.preprocessing.normalizer import preprocess_audio
from src.preprocessing.vad import trim_silence
from src.features.mfcc_extractor import extract_mfcc_with_deltas, aggregate_features
from src.features.feature_normalizer import apply_cmvn
from src.models.hmm_model import HMMClassifier
from src.models.classifier import load_model


class VoiceRecognizer:
    """Clase para reconocimiento de voz."""

    def __init__(self, model_type: str = "svm"):
        """
        Args:
            model_type: Tipo de modelo ("hmm", "svm", "rf", "mlp")
        """
        self.model_type = model_type
        self.model = None
        self.is_hmm = model_type == "hmm"

    def load_model(self, model_path: str = None):
        """Carga el modelo desde disco."""
        if model_path is None:
            if self.is_hmm:
                model_path = str(MODELS_DIR / "hmm_model.pkl")
            else:
                model_path = str(MODELS_DIR / f"{self.model_type}_model.pkl")

        if self.is_hmm:
            self.model = HMMClassifier()
            self.model.load(model_path)
        else:
            self.model = load_model(model_path)

        print(f"Modelo cargado: {model_path}")

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocesa el audio."""
        # Normalizar y aplicar pre-énfasis
        audio = preprocess_audio(audio)
        # Recortar silencios
        audio = trim_silence(audio, sr)
        return audio

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extrae características del audio."""
        # MFCCs con deltas
        features = extract_mfcc_with_deltas(audio)
        # Normalización CMVN
        features = apply_cmvn(features)
        return features

    def predict(self, audio_path: str) -> tuple:
        """
        Predice la palabra para un archivo de audio.

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Tuple (palabra_predicha, confianza)
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load_model() primero.")

        # Cargar audio
        audio, sr = load_audio(audio_path)
        if audio is None:
            return None, 0.0

        # Preprocesar
        audio = self.preprocess_audio(audio, sr)

        if len(audio) < sr * 0.1:  # Muy corto
            return None, 0.0

        # Extraer características
        features = self.extract_features(audio)

        # Predecir
        if self.is_hmm:
            prediction = self.model.predict(features)
            scores = self.model.predict_proba(features)
            # Normalizar scores para obtener "confianza"
            max_score = max(scores.values())
            confidence = 1.0  # HMM no da probabilidades normalizadas directamente
        else:
            features_agg = aggregate_features(features).reshape(1, -1)
            prediction = self.model.predict(features_agg)[0]
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features_agg)[0]
                    confidence = max(proba)
                except:
                    confidence = 1.0
            else:
                confidence = 1.0

        return prediction, confidence

    def predict_top_k(self, audio_path: str, k: int = 3) -> list:
        """
        Retorna las top-k predicciones.

        Args:
            audio_path: Ruta al archivo de audio
            k: Número de predicciones

        Returns:
            Lista de tuplas (palabra, confianza)
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado.")

        # Cargar y preprocesar
        audio, sr = load_audio(audio_path)
        if audio is None:
            return []

        audio = self.preprocess_audio(audio, sr)
        if len(audio) < sr * 0.1:
            return []

        features = self.extract_features(audio)

        if self.is_hmm:
            scores = self.model.predict_proba(features)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [(word, score) for word, score in sorted_scores[:k]]
        else:
            features_agg = aggregate_features(features).reshape(1, -1)
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features_agg)[0]
                    classes = self.model.classes_
                    sorted_idx = np.argsort(proba)[::-1][:k]
                    return [(classes[i], proba[i]) for i in sorted_idx]
                except:
                    pass

            prediction = self.model.predict(features_agg)[0]
            return [(prediction, 1.0)]


def extract_expected_word(filename: str) -> str:
    """Extrae la palabra esperada del nombre del archivo (ej: 'Agua1.wav' -> 'agua')."""
    import re
    name = Path(filename).stem  # Sin extensión
    # Quitar números del final
    word = re.sub(r'\d+$', '', name)
    return word.lower()


def evaluate_folder(folder_path: Path, recognizer: VoiceRecognizer):
    """Evalúa todos los audios en una carpeta automáticamente."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    audio_files = list(folder_path.glob("*.wav")) + list(folder_path.glob("*.mp3"))

    if not audio_files:
        print(f"No se encontraron archivos de audio en: {folder_path}")
        return

    print(f"\nEvaluando {len(audio_files)} archivos...\n")

    results = []
    correct = 0
    total = 0

    # Para matriz de confusión
    all_expected = []
    all_predicted = []

    for audio_file in sorted(audio_files):
        expected = extract_expected_word(audio_file.name)
        predicted, confidence = recognizer.predict(str(audio_file))

        if predicted:
            predicted = predicted.lower()
            is_correct = expected == predicted
            if is_correct:
                correct += 1
            total += 1
            results.append({
                'archivo': audio_file.name,
                'esperado': expected,
                'prediccion': predicted,
                'correcto': is_correct
            })
            all_expected.append(expected)
            all_predicted.append(predicted)
        else:
            total += 1
            results.append({
                'archivo': audio_file.name,
                'esperado': expected,
                'prediccion': '(error)',
                'correcto': False
            })
            all_expected.append(expected)
            all_predicted.append('error')

    accuracy = (correct / total * 100) if total > 0 else 0

    # Crear figura con subplots
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f'Evaluación de Reconocimiento de Voz - Modelo: {recognizer.model_type.upper()}',
                 fontsize=14, fontweight='bold')

    # 1. Tabla de resultados (izquierda)
    ax1 = plt.subplot(1, 2, 1)
    ax1.axis('off')

    # Preparar datos para tabla
    table_data = []
    colors = []
    for r in results:
        status = '✓' if r['correcto'] else '✗'
        table_data.append([r['archivo'], r['esperado'], r['prediccion'], status])
        colors.append(['white', 'white',
                      '#90EE90' if r['correcto'] else '#FFB6C1',
                      '#90EE90' if r['correcto'] else '#FFB6C1'])

    table = ax1.table(
        cellText=table_data,
        colLabels=['Archivo', 'Esperado', 'Predicción', ''],
        cellColours=colors,
        colColours=['#4472C4', '#4472C4', '#4472C4', '#4472C4'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Colorear encabezados
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color='white', fontweight='bold')

    # 2. Gráfica de resultados (derecha superior)
    ax2 = plt.subplot(2, 2, 2)

    labels = ['Correctos', 'Incorrectos']
    sizes = [correct, total - correct]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11})
    ax2.set_title(f'Precisión: {correct}/{total} ({accuracy:.1f}%)', fontsize=12, fontweight='bold')

    # 3. Resultados por palabra (derecha inferior)
    ax3 = plt.subplot(2, 2, 4)

    # Contar aciertos por palabra
    word_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        word_stats[r['esperado']]['total'] += 1
        if r['correcto']:
            word_stats[r['esperado']]['correct'] += 1

    words = list(word_stats.keys())
    correct_counts = [word_stats[w]['correct'] for w in words]
    total_counts = [word_stats[w]['total'] for w in words]
    incorrect_counts = [t - c for t, c in zip(total_counts, correct_counts)]

    x = np.arange(len(words))
    width = 0.6

    bars1 = ax3.bar(x, correct_counts, width, label='Correctos', color='#2ecc71')
    bars2 = ax3.bar(x, incorrect_counts, width, bottom=correct_counts, label='Incorrectos', color='#e74c3c')

    ax3.set_ylabel('Cantidad')
    ax3.set_title('Resultados por Palabra', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(words, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, max(total_counts) + 1)

    # Añadir etiquetas en las barras
    for i, (c, t) in enumerate(zip(correct_counts, total_counts)):
        ax3.annotate(f'{c}/{t}', xy=(i, t), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Guardar y mostrar
    output_path = folder_path.parent / 'results' / 'evaluacion_audios.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*60}")
    print(f"  RESULTADOS DE EVALUACIÓN")
    print(f"{'='*60}")
    print(f"  Modelo: {recognizer.model_type.upper()}")
    print(f"  Archivos evaluados: {total}")
    print(f"  Correctos: {correct}")
    print(f"  Incorrectos: {total - correct}")
    print(f"  Precisión: {accuracy:.1f}%")
    print(f"{'='*60}")
    print(f"\n  Gráfica guardada en: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predecir palabra en audio')
    parser.add_argument('audio', type=str, help='Ruta al archivo de audio o carpeta')
    parser.add_argument('--model', type=str, default='svm',
                       choices=['hmm', 'svm', 'rf', 'mlp'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--top-k', type=int, default=1,
                       help='Mostrar top-k predicciones')
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: No se encontró: {audio_path}")
        sys.exit(1)

    print("=" * 60)
    print("RECONOCIMIENTO DE VOZ")
    print("=" * 60)

    # Crear reconocedor
    recognizer = VoiceRecognizer(model_type=args.model)

    # Cargar modelo
    try:
        recognizer.load_model()
    except FileNotFoundError:
        print(f"\nERROR: No se encontró el modelo {args.model}.")
        print("Ejecuta primero: python scripts/train.py")
        sys.exit(1)

    # Si es carpeta, evaluar todos los archivos
    if audio_path.is_dir():
        evaluate_folder(audio_path, recognizer)
    else:
        # Predecir archivo individual
        print(f"\nAnalizando: {audio_path}")
        print("-" * 40)

        if args.top_k > 1:
            predictions = recognizer.predict_top_k(str(audio_path), k=args.top_k)
            print(f"\nTop-{args.top_k} Predicciones:")
            for i, (word, confidence) in enumerate(predictions, 1):
                print(f"  {i}. {word}")
        else:
            word, confidence = recognizer.predict(str(audio_path))
            if word:
                print(f"\nPalabra detectada: {word}")
            else:
                print("\nNo se pudo detectar ninguna palabra.")

    print()


if __name__ == "__main__":
    main()
