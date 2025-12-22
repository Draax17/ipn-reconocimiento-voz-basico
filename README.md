# Sistema de Reconocimiento de Voz Básico

**Proyecto Final de Reconocimiento de Voz - ESCOM-IPN**

Implementación de un sistema de reconocimiento de voz desde cero mediante técnicas de procesamiento digital de señales y aprendizaje automático.

## 📋 Descripción

Este proyecto busca implementar un sistema de reconocimiento de voz desde cero demostrando los fundamentos del procesamiento de señales de audio y modelos estadísticos/clásicos de machine learning. A diferencia de las APIs comerciales como Whisper, Google Speech-to-Text o Amazon Transcribe, este proyecto revela la lógica interna del procesamiento y permite un aprendizaje académico profundo de los pasos fundamentales del reconocimiento automático del habla (ASR).

## 🎯 Objetivo General

Diseñar e implementar un sistema básico de reconocimiento de voz que convierta fragmentos de audio en texto, utilizando técnicas propias de procesamiento de señales y modelos de clasificación, sin emplear APIs de terceros.

## 🎯 Objetivos Específicos

- Preprocesar señales de audio eliminando ruido y normalizando amplitud
- Implementar extracción de características acústicas como MFCCs (Mel-Frequency Cepstral Coefficients) o espectrogramas log-mel
- Entrenar un modelo de reconocimiento usando algoritmos clásicos (HMM, GMM, SVM o redes neuronales simples)
- Implementar un módulo de evaluación para medir precisión de reconocimiento frente a un corpus de prueba
- Documentar la arquitectura y resultados para fines académicos

## 📊 Alcance del Proyecto

- **Vocabulario limitado**: 10–20 palabras o frases
- **Entrenamiento**: Conjunto de audios recolectados por los estudiantes o mediante corpus libres (TIMIT, LibriSpeech reducido)
- **Limitaciones**: No se implementará un modelo de lenguaje complejo ni un sistema de reconocimiento a gran escala
- **Requisito**: Funcionamiento en PC sin depender de servicios en la nube ni APIs comerciales

## 🔧 Tecnologías Utilizadas

### Lenguajes
- **Python 3.x**

### Librerías Principales
- **NumPy**: Operaciones numéricas y manejo de arrays
- **SciPy**: Procesamiento de señales y funciones científicas
- **librosa**: Análisis de audio y extracción de características
- **PyTorch/TensorFlow**: Entrenamiento de modelos de machine learning
- **scikit-learn**: Algoritmos clásicos de ML (HMM, GMM, SVM, etc.)
- **matplotlib**: Visualización de resultados

### Corpus de Datos
- Corpus libres para pruebas (TIMIT, LibriSpeech reducido, o datos propios)

## 🚫 Restricciones

**No está permitido el uso de:**
- APIs externas de reconocimiento de voz (Whisper, Google Speech-to-Text, Amazon Transcribe, etc.)
- Modelos preentrenados cerrados

## 📁 Estructura del Proyecto

```
ipn-reconocimiento-voz-basico/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/                          # Código fuente
│   ├── preprocessing/            # Módulo de preprocesamiento
│   │   ├── __init__.py
│   │   ├── audio_processor.py   # Conversión, normalización, filtrado
│   │   └── noise_reduction.py   # Eliminación de ruido
│   │
│   ├── features/                 # Extracción de características
│   │   ├── __init__.py
│   │   ├── mfcc_extractor.py    # Extracción de MFCCs
│   │   └── spectrogram.py       # Espectrogramas log-mel
│   │
│   ├── models/                   # Modelos de ML
│   │   ├── __init__.py
│   │   ├── hmm_gmm.py           # Modelo HMM/GMM
│   │   ├── svm_classifier.py    # Clasificador SVM
│   │   └── neural_network.py    # Red neuronal básica
│   │
│   ├── recognition/              # Módulo de reconocimiento
│   │   ├── __init__.py
│   │   └── predictor.py         # Predicción de palabras/frases
│   │
│   └── evaluation/               # Evaluación
│       ├── __init__.py
│       ├── metrics.py           # Cálculo de métricas (accuracy, WER)
│       └── validator.py         # Validación cruzada
│
├── data/                         # Datos del proyecto
│   ├── raw/                      # Audios originales
│   ├── processed/                # Audios preprocesados
│   └── corpus/                   # Corpus de entrenamiento
│
├── models/                       # Modelos entrenados guardados
│
├── results/                      # Resultados y visualizaciones
│   ├── metrics/                  # Métricas de evaluación
│   └── plots/                    # Gráficas y visualizaciones
│
├── docs/                         # Documentación
│   └── informe_tecnico.pdf       # Documento técnico final
│
└── tests/                        # Pruebas unitarias
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_models.py
```

## 🔬 Metodología

### 1. Recolección de Datos
- Captura de muestras de voz de los integrantes del equipo
- Uso de un corpus reducido de acceso libre (TIMIT, LibriSpeech reducido)

### 2. Preprocesamiento
- Conversión a mono y 16 kHz
- Eliminación de ruido con filtros digitales (filtro de Wiener o reducción espectral)
- Normalización de amplitud

### 3. Extracción de Características
- Implementación de MFCCs usando librerías científicas (NumPy, SciPy, librosa)
- Representación de cada señal de audio como un vector de características

### 4. Modelado y Entrenamiento
- Implementación de un modelo clásico de reconocimiento:
  - **HMM/GMM**: Algoritmos de probabilidad secuencial
  - **Alternativamente**: Red neuronal feedforward o RNN básica entrenada en PyTorch/TensorFlow (sin uso de APIs preentrenadas)

### 5. Reconocimiento
- Módulo que recibe un audio desconocido, extrae características y predice la palabra/frase más probable

### 6. Evaluación
- **Métricas**: Tasa de aciertos (accuracy), tasa de error de palabra (WER)
- Comparación con distintos modelos entrenados

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd ipn-reconocimiento-voz-basico
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Entrenamiento del Modelo

```bash
python src/models/train.py --data_path data/corpus --model_type svm --output models/
```

### Reconocimiento de Audio

```bash
python src/recognition/predictor.py --audio_path audio_test.wav --model_path models/model.pkl
```

### Evaluación

```bash
python src/evaluation/evaluator.py --test_data data/test --model_path models/model.pkl
```

## 📈 Resultados Esperados

- Prototipo funcional capaz de reconocer un conjunto limitado de palabras/frases
- Análisis comparativo entre distintos métodos de modelado
- Documento técnico con fundamentos teóricos, diseño, pruebas y conclusiones
- Presentación en PowerPoint o similar (opcional)

## 📚 Documentación Técnica

El documento técnico final debe contener las siguientes secciones:

1. **Portada**
2. **Índice**
3. **Introducción**
4. **Estado del Arte**
5. **Desarrollo**
6. **Conclusiones**
7. **Referencias**

## 🎓 Impacto Académico

Este proyecto permitirá a los estudiantes:

- Comprender a fondo el flujo de un sistema de reconocimiento de voz
- Poner en práctica conocimientos de matemáticas, estadística, programación y machine learning
- Prepararse para proyectos más complejos en el área de inteligencia artificial y procesamiento del lenguaje natural

## 👥 Contribuidores

- [Lista de integrantes del equipo]

## 📄 Licencia

Este proyecto es desarrollado con fines académicos para el curso de Reconocimiento de Voz en ESCOM-IPN.

## 📝 Referencias

- [Agregar referencias bibliográficas relevantes]

---

**Instituto Politécnico Nacional**  
**Escuela Superior de Cómputo (ESCOM)**  
**Proyecto Final de Reconocimiento de Voz**

