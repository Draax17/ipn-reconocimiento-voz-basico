# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic voice recognition project for ESCOM-IPN implementing vowel detection and classification from scratch using signal processing and classical machine learning. The system processes Spanish vowel audio samples (a, e, i, o, u) without external speech recognition APIs.

## Running the Application

```bash
# Run from the "Deteccion de Vocales" directory
cd "Deteccion de Vocales"
python main.py
```

The script expects a `dataset/` folder with subdirectories for each vowel class (a, e, i, o, u, noise), each containing `.wav` files. Output goes to `resultados/`.

## Dependencies

Core libraries (not in requirements.txt yet):
- `parselmouth` - Python interface to Praat for acoustic analysis
- `numpy` - Numerical operations
- `scikit-learn` - DecisionTreeClassifier, cross-validation, metrics
- `matplotlib` - Visualization

Install with: `pip install praat-parselmouth numpy scikit-learn matplotlib`

## Architecture

The project is a single-file implementation (`Deteccion de Vocales/main.py`) with these components:

### Feature Extraction
- **Formant extraction**: F1, F2, F3 frequencies via Praat/Parselmouth's Burg method
- **Energy analysis**: Per-window energy in dB for voice activity detection
- **Spectral band energy**: Low (300-1000 Hz) vs high (3000-5500 Hz) frequency comparison

### Classification Pipeline
1. **Segment detection**: Energy-based voice activity detection with -40 dB threshold
2. **Phonetic classification**: Distinguishes voiced/unvoiced, vowels, fricatives, stops based on F0 and spectral features
3. **Vowel classification**: Decision tree classifier using F1, F2, and duration features with 5-fold cross-validation

### Key Functions
- `extraer_caracteristicas_segmentos()` - Extracts F1, F2, duration from sounding segments
- `clasificar_segmento()` - Classifies segment type (vocal, fricativa_sonora, parada_sonora, etc.)
- `clasificar_ventanas()` - Frame-level silence/voice/unvoice classification
- `detectar_segmentos_por_energia()` - Energy-based speech segment detection

### Output
- `resultados/metricas_vocales.txt` - Multi-class vowel classification metrics
- `resultados/metricas_vocal_no_vocal.txt` - Binary vocal/non-vocal metrics
- `resultados/malla_fonetica.png` - F1-F2 phonetic vowel chart
- Per-audio PNG files with formants, energy contours, and waveform classification

## Constraints

- No external speech recognition APIs (Whisper, Google STT, etc.)
- No pre-trained closed models
- Must run locally without cloud services
