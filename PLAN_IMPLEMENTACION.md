# Plan de Implementación: Sistema de Reconocimiento de Voz

## Resumen del Proyecto

**Objetivo:** Implementar un sistema de reconocimiento de voz desde cero que convierta fragmentos de audio en texto, utilizando técnicas de procesamiento digital de señales y modelos clásicos de machine learning.

**Alcance:** Reconocimiento de un vocabulario reducido (~15-20 palabras seleccionadas del dataset).

**Restricciones:**
- Sin APIs externas (Whisper, Google STT, etc.)
- Sin modelos preentrenados cerrados
- Ejecución local sin servicios en la nube

---

## Recursos Disponibles

### Dataset
- **Ubicación:** `dataset_filtrado/dataset_filtrado/`
- **Formato:** Archivos `.opus` del corpus Common Voice (Mozilla)
- **Contenido:** 103 palabras diferentes, ~750+ muestras por palabra
- **Idioma:** Español

### Código Base
- **Ubicación:** `Deteccion de Vocales/main.py`
- **Funcionalidades reutilizables:**
  - Extracción de formantes (F1, F2, F3) con Parselmouth/Praat
  - Detección de segmentos por energía
  - Clasificación de ventanas (silence/voice/unvoice)
  - Validación cruzada con scikit-learn

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE RECONOCIMIENTO                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────┐ │
│  │  Audio   │───▶│Preprocesado  │───▶│ Extracción │───▶│ Modelo  │ │
│  │  Input   │    │              │    │ Features   │    │   ML    │ │
│  │  (.opus) │    │ - Conversión │    │ - MFCCs    │    │         │ │
│  └──────────┘    │ - Normalizar │    │ - Delta    │    │ HMM/GMM │ │
│                  │ - Reducción  │    │ - Energía  │    │   o     │ │
│                  │   de ruido   │    │            │    │  RNN    │ │
│                  └──────────────┘    └────────────┘    └────┬────┘ │
│                                                              │      │
│                                                              ▼      │
│                                                        ┌─────────┐  │
│                                                        │ Palabra │  │
│                                                        │Predicha │  │
│                                                        └─────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Fases de Implementación

### Fase 1: Preparación del Entorno y Datos

#### 1.1 Configuración del Entorno
- [ ] Crear `requirements.txt` con dependencias:
  ```
  numpy
  scipy
  librosa
  scikit-learn
  matplotlib
  praat-parselmouth
  soundfile
  pydub
  hmmlearn        # Para HMM/GMM
  torch           # Opcional: para RNN
  torchaudio      # Opcional: para RNN
  ```
- [ ] Crear estructura de carpetas del proyecto:
  ```
  proyecto/
  ├── src/
  │   ├── preprocessing/
  │   ├── features/
  │   ├── models/
  │   └── evaluation/
  ├── data/
  │   ├── raw/           # Dataset original
  │   ├── processed/     # Audio preprocesado
  │   └── features/      # Features extraídas
  ├── models/            # Modelos entrenados
  ├── results/           # Métricas y gráficas
  └── notebooks/         # Experimentación
  ```

#### 1.2 Selección del Vocabulario
- [ ] Seleccionar 15-20 palabras del dataset considerando:
  - Diversidad fonética (diferentes sonidos iniciales/finales)
  - Cantidad suficiente de muestras (>500 por palabra)
  - Longitud variada (monosílabas, bisílabas, trisílabas)

**Palabras sugeridas (basadas en el dataset):**
| Palabra | Muestras | Características |
|---------|----------|-----------------|
| agua | ~750 | Bisílaba, vocal inicial |
| con | ~750 | Monosílaba, consonante |
| del | ~750 | Monosílaba |
| dos | ~750 | Monosílaba |
| entre | ~750 | Trisílaba |
| esta | ~750 | Bisílaba |
| fue | ~750 | Monosílaba |
| gran | ~750 | Monosílaba |
| las | ~750 | Monosílaba |
| los | ~750 | Monosílaba |
| más | ~750 | Monosílaba |
| muy | ~750 | Monosílaba |
| para | ~750 | Bisílaba |
| por | ~750 | Monosílaba |
| que | ~750 | Monosílaba |
| son | ~750 | Monosílaba |
| una | ~750 | Bisílaba |
| vida | ~750 | Bisílaba |

#### 1.3 Conversión de Formato
- [ ] Implementar script para convertir `.opus` a `.wav`:
  - Frecuencia de muestreo: 16 kHz
  - Canales: Mono
  - Bits: 16-bit PCM

---

### Fase 2: Preprocesamiento de Señales

#### 2.1 Normalización de Amplitud
- [ ] Implementar normalización peak a [-1, 1]
- [ ] Implementar normalización RMS para volumen consistente

#### 2.2 Reducción de Ruido
- [ ] Implementar filtro paso-banda (80 Hz - 8000 Hz)
- [ ] Implementar reducción espectral simple (spectral subtraction)
- [ ] Alternativa: Filtro de Wiener básico

#### 2.3 Detección de Actividad de Voz (VAD)
- [ ] Reutilizar `detectar_segmentos_por_energia()` de la base
- [ ] Ajustar umbrales para el nuevo dataset
- [ ] Recortar silencios al inicio y final

#### 2.4 Pre-énfasis
- [ ] Implementar filtro de pre-énfasis: y[n] = x[n] - α·x[n-1], α ≈ 0.97

---

### Fase 3: Extracción de Características

#### 3.1 MFCCs (Características Principales)
- [ ] Implementar extracción de MFCCs con librosa:
  ```python
  # Parámetros sugeridos
  n_mfcc = 13          # Número de coeficientes
  n_fft = 512          # Tamaño de ventana FFT (~32ms a 16kHz)
  hop_length = 160     # Salto de ventana (~10ms)
  n_mels = 40          # Filtros mel
  fmin = 80            # Frecuencia mínima
  fmax = 8000          # Frecuencia máxima
  ```

#### 3.2 Características Delta
- [ ] Calcular derivadas de primer orden (delta)
- [ ] Calcular derivadas de segundo orden (delta-delta)
- [ ] Vector final: 39 dimensiones (13 MFCC + 13 delta + 13 delta-delta)

#### 3.3 Características Complementarias (opcional)
- [ ] Energía por frame (log-energy)
- [ ] Zero Crossing Rate (ZCR)
- [ ] Formantes F1, F2 (reutilizar código existente)

#### 3.4 Normalización de Features
- [ ] Implementar Cepstral Mean Normalization (CMN)
- [ ] Implementar Cepstral Variance Normalization (CVN)

---

### Fase 4: Modelado y Entrenamiento

#### Opción A: HMM-GMM (Recomendado para enfoque clásico)

##### 4A.1 Arquitectura HMM
- [ ] Implementar modelo HMM por palabra usando `hmmlearn`:
  ```python
  # Parámetros sugeridos
  n_states = 5-8       # Estados por palabra (depende de longitud)
  n_components = 8-16  # Componentes GMM por estado
  covariance_type = 'diag'
  n_iter = 100
  ```

##### 4A.2 Entrenamiento
- [ ] Entrenar un modelo HMM por cada palabra del vocabulario
- [ ] Implementar inicialización con k-means
- [ ] Usar algoritmo Baum-Welch (EM) para entrenamiento

##### 4A.3 Reconocimiento
- [ ] Implementar decodificación Viterbi
- [ ] Clasificar por máxima verosimilitud entre modelos

#### Opción B: Red Neuronal Simple (Alternativa)

##### 4B.1 Arquitectura RNN/LSTM
- [ ] Implementar red con PyTorch:
  ```python
  # Arquitectura sugerida
  LSTM(input_size=39, hidden_size=128, num_layers=2, bidirectional=True)
  Linear(256, n_classes)  # n_classes = número de palabras
  ```

##### 4B.2 Entrenamiento
- [ ] Implementar padding/truncating para secuencias
- [ ] Usar CrossEntropyLoss
- [ ] Optimizador: Adam, lr=0.001
- [ ] Batch size: 32-64

#### Opción C: Clasificador Clásico con Features Agregadas

##### 4C.1 Agregación de Features
- [ ] Calcular estadísticas por utterance:
  - Media y desviación estándar de MFCCs
  - Min/Max de cada coeficiente
  - Pendiente temporal

##### 4C.2 Clasificadores
- [ ] SVM con kernel RBF
- [ ] Random Forest
- [ ] Gradient Boosting (XGBoost/LightGBM)

---

### Fase 5: División de Datos y Validación

#### 5.1 Partición del Dataset
- [ ] Dividir datos:
  - Entrenamiento: 70%
  - Validación: 15%
  - Prueba: 15%
- [ ] Asegurar estratificación por clase
- [ ] Evitar data leakage (mismos hablantes en train/test si es posible)

#### 5.2 Validación Cruzada
- [ ] Implementar k-fold cross-validation (k=5)
- [ ] Calcular intervalos de confianza

---

### Fase 6: Evaluación

#### 6.1 Métricas de Clasificación
- [ ] Accuracy (exactitud global)
- [ ] Precision por clase
- [ ] Recall por clase
- [ ] F1-Score (macro y weighted)
- [ ] Matriz de confusión

#### 6.2 Métricas de Reconocimiento de Voz
- [ ] Word Error Rate (WER) - aunque para palabras aisladas es equivalente a (1 - accuracy)
- [ ] Top-k accuracy (k=3, k=5)

#### 6.3 Análisis de Errores
- [ ] Identificar pares de palabras más confundidas
- [ ] Análisis por longitud de palabra
- [ ] Análisis por características fonéticas

#### 6.4 Visualizaciones
- [ ] Matriz de confusión (heatmap)
- [ ] Curvas de aprendizaje
- [ ] t-SNE/UMAP de features para ver separabilidad de clases
- [ ] Espectrogramas y MFCCs de ejemplos

---

### Fase 7: Módulo de Predicción en Tiempo Real

#### 7.1 Pipeline de Inferencia
- [ ] Implementar función `predecir_palabra(audio_path)`:
  1. Cargar audio
  2. Preprocesar
  3. Extraer features
  4. Clasificar
  5. Retornar palabra y confianza

#### 7.2 Interfaz de Usuario (Opcional)
- [ ] CLI básica para probar con archivos de audio
- [ ] Grabación en tiempo real con micrófono (opcional)

---

### Fase 8: Documentación

#### 8.1 Documento Técnico
- [ ] **Portada:** Título, autores, institución, fecha
- [ ] **Índice:** Secciones numeradas
- [ ] **Introducción:**
  - Contexto del reconocimiento de voz
  - Motivación del proyecto
  - Objetivos
- [ ] **Estado del Arte:**
  - Historia del ASR
  - Técnicas clásicas (HMM, GMM, DTW)
  - Técnicas modernas (DNN, RNN, Transformers)
  - Justificación de las técnicas elegidas
- [ ] **Desarrollo:**
  - Descripción del dataset
  - Pipeline de preprocesamiento
  - Extracción de características (teoría MFCCs)
  - Modelo de clasificación
  - Implementación técnica
- [ ] **Resultados:**
  - Métricas obtenidas
  - Comparación entre modelos (si aplica)
  - Análisis de errores
- [ ] **Conclusiones:**
  - Logros alcanzados
  - Limitaciones
  - Trabajo futuro
- [ ] **Referencias:** Formato APA/IEEE

---

## Dependencias entre Fases

```
Fase 1 ──▶ Fase 2 ──▶ Fase 3 ──▶ Fase 4
                         │
                         ▼
                      Fase 5 ──▶ Fase 6
                                   │
                                   ▼
                                Fase 7
                                   │
                                   ▼
                                Fase 8
```

---

## Riesgos y Mitigaciones

| Riesgo | Impacto | Mitigación |
|--------|---------|------------|
| Dataset muy grande para procesar | Alto | Submuestrear a 200-300 audios por palabra |
| Archivos .opus no soportados | Medio | Convertir a WAV con ffmpeg/pydub |
| HMM no converge | Medio | Probar diferentes inicializaciones, usar clasificador clásico como alternativa |
| Baja precisión (<70%) | Alto | Aumentar features, probar otros modelos, reducir vocabulario |
| Overfitting | Medio | Regularización, dropout, early stopping |

---

## Métricas de Éxito

| Métrica | Objetivo Mínimo | Objetivo Deseable |
|---------|-----------------|-------------------|
| Accuracy | 70% | >85% |
| F1-Score (macro) | 0.65 | >0.80 |
| Tiempo de inferencia | <2s por audio | <0.5s por audio |

---

## Estructura de Archivos Propuesta

```
ipn-reconocimiento-voz-basico/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Parámetros globales
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_converter.py    # Conversión opus→wav
│   │   ├── normalizer.py         # Normalización
│   │   ├── noise_reduction.py    # Reducción de ruido
│   │   └── vad.py                # Voice Activity Detection
│   ├── features/
│   │   ├── __init__.py
│   │   ├── mfcc_extractor.py     # Extracción de MFCCs
│   │   ├── delta_features.py     # Deltas
│   │   └── feature_normalizer.py # CMN/CVN
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hmm_model.py          # Modelo HMM-GMM
│   │   ├── rnn_model.py          # Modelo RNN (opcional)
│   │   └── classifier.py         # Clasificadores clásicos
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Cálculo de métricas
│   │   └── visualizations.py     # Gráficas
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py        # Carga de datos
├── scripts/
│   ├── prepare_data.py           # Preparación del dataset
│   ├── train.py                  # Entrenamiento
│   ├── evaluate.py               # Evaluación
│   └── predict.py                # Predicción
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_comparison.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
├── results/
├── docs/
│   └── documento_tecnico.md
├── requirements.txt
├── README.md
└── main.py                       # Punto de entrada principal
```

---

## Próximos Pasos Inmediatos

1. **Crear `requirements.txt`** con todas las dependencias
2. **Implementar script de conversión** `.opus` → `.wav`
3. **Seleccionar las 15-20 palabras** finales del vocabulario
4. **Crear el módulo de extracción de MFCCs**
5. **Entrenar un modelo baseline** (SVM o Random Forest) para validar el pipeline

---

## Referencias Técnicas Sugeridas

1. Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition.
2. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
3. Graves, A. et al. (2013). Speech recognition with deep recurrent neural networks.
4. Librosa documentation: https://librosa.org/doc/
5. hmmlearn documentation: https://hmmlearn.readthedocs.io/
