# Clasificador de Segmento de Precio de Vehículos Usados

Proyecto de Aprendizaje de Máquinas. La solución clasifica vehículos usados en tres segmentos de precio —**Económico**, **Medio** y **Premium**— utilizando sus características técnicas y comerciales.

La pregunta que orienta el proyecto es:
> ¿Qué características de un vehículo usado permiten clasificarlo en un segmento de precio y qué estrategia de modelado ofrece el mejor equilibrio de desempeño entre las tres clases?

## Integrantes

- Gabriel Jara Rivas
- Matías Lillo Yévenes
- Nicolás Duarte Maldonado
- Nicolás Navarrete Caro

## Descripción del problema

A partir del dataset `vehicles_us.csv`, el proyecto formula una tarea de clasificación multiclase. La variable objetivo se construye utilizando los siguientes umbrales:

| Segmento  | Definición                          |
| --------- | ------------------------------------ |
| Económico | Precio inferior a USD 5.000          |
| Medio     | Precio entre USD 5.000 y USD 20.000  |
| Premium   | Precio superior a USD 20.000         |

Después del proceso de limpieza, el conjunto utilizado para el modelado contiene **50.267 vehículos**, distribuidos de la siguiente manera:

| Segmento  | Cantidad   | Porcentaje  |
| --------- | ---------- | ----------- |
| Económico | 11.746     | 23,4 %      |
| Medio     | 29.856     | 59,4 %      |
| Premium   | 8.665      | 17,2 %      |
| **Total** | **50.267** | **100,0 %** |

Esta distribución presenta un desbalance relevante, principalmente por el predominio de la clase Medio. Por esta razón, la métrica principal del proyecto es el **F1-macro**, complementada con accuracy, precisión, recall por clase y AUC multiclase One-vs-Rest con promedio macro.

## Resultado del modelo final

El modelo seleccionado es un pipeline compuesto por:

1. Preprocesamiento de variables numéricas y categóricas (sin la variable `age`, redundante con `model_year`).
2. Balanceo de clases mediante SMOTE.
3. Clasificación mediante Random Forest de 200 árboles.

El modelo se entrenó utilizando los conjuntos de entrenamiento y validación y posteriormente se evaluó sobre **7.541 vehículos del conjunto de test**, los cuales no fueron utilizados durante el entrenamiento, la validación ni la selección de hiperparámetros.

### Resultados por segmento

| Segmento       | Precisión | Recall | F1-score | Soporte |
| -------------- | --------- | ------ | -------- | ------- |
| Económico      | 0,797     | 0,791  | 0,794    | 1.762   |
| Medio          | 0,883     | 0,883  | 0,883    | 4.479   |
| Premium        | 0,866     | 0,875  | 0,871    | 1.300   |
| Global (macro) | 0,849     | 0,850  | 0,849    | —       |

### Resultados globales

- **Accuracy:** 0,860.
- **F1-macro:** 0,849.
- **AUC-OVR macro:** 0,953.
- **Casos de test:** 7.541.
- **Tamaño comprimido del modelo:** aproximadamente 66,3 MB.

El AUC-OVR macro de 0,953 evidencia una alta capacidad del modelo para discriminar cada segmento frente a los restantes mediante las probabilidades estimadas. Sin embargo, las probabilidades no han sido sometidas a un procedimiento específico de calibración, por lo que deben interpretarse como estimaciones orientativas y no como niveles de confianza perfectamente calibrados.

Las métricas corresponden al dataset y al mercado analizado, por lo que no garantizan el mismo desempeño en otros países, periodos, plataformas o fuentes de información.

## Comparación de técnicas bajo un protocolo homogéneo

Además de la comparación exploratoria inicial entre Regresión Logística, Random Forest, XGBoost y tres arquitecturas de Deep Learning, se realizó una segunda comparación aplicando SMOTE de forma **uniforme** (sin mezclar `class_weight`) sobre las cinco técnicas cuya representación de entrada es compatible con el remuestreo sintético:

| Modelo               | Accuracy | F1-macro |
| --------------------- | -------- | -------- |
| Regresión Logística    | 0,767    | 0,759    |
| Random Forest          | 0,860    | 0,848    |
| XGBoost                | 0,846    | 0,831    |
| MLP Shallow            | 0,806    | 0,797    |
| MLP Profunda           | 0,803    | 0,796    |

Random Forest se mantiene como la técnica de mejor desempeño bajo este protocolo controlado. Ver `src/homogeneous_comparison.py` y `reports/homogeneous_comparison_results.csv`.

(La arquitectura MLP con Embeddings queda fuera de esta comparación específica porque sus variables categóricas se representan como índices enteros, no como vectores one-hot, y SMOTE no puede generar índices sintéticos categóricos válidos.)

## Matriz de confusión

| Segmento real | Económico | Medio | Premium |
| ------------- | --------- | ----- | ------- |
| Económico     | 1.393     | 365   | 4       |
| Medio         | 351       | 3.956 | 172     |
| Premium       | 3         | 159   | 1.138   |

La mayoría de los errores ocurre entre segmentos vecinos. Las confusiones extremas entre Económico y Premium suman solo siete casos al considerar ambas direcciones.

## Explicabilidad

El análisis de explicabilidad (global y local) se calcula directamente sobre el Random Forest final de 200 árboles —el mismo objeto serializado en `models/final_model.joblib` que expone la API—, sin modelo sustituto de por medio.

Las variables globalmente más relevantes son:

1. `model_year` (año del modelo)
2. `odometer` (kilometraje)
3. `is_4wd` (tracción 4×4)
4. `cylinders` (cantidad de cilindros)
5. `condition_ord` (condición del vehículo)

Al no incluir la variable `age` (redundante con `model_year`, ya que `age = 2019 - model_year`), el ranking de importancia no reparte artificialmente el crédito de la antigüedad del vehículo entre dos columnas equivalentes.

**Explicación local en la API.** Cada respuesta del endpoint `/predict` incluye además una explicación calculada en el momento de la solicitud, mediante `shap.TreeExplainer`, sobre la predicción específica de ese vehículo. Esto añade aproximadamente 2,6 segundos de latencia por solicitud (el explicador se construye una sola vez al iniciar la aplicación, pero el cálculo de valores de Shapley sobre un bosque de 200 árboles sigue siendo el paso más costoso).

## Estructura del repositorio

```
acif104_s9_equipo4/
├── backend/
│   └── app.py
├── data/
│   └── vehicles_us.csv
├── docs/
│   └── acif104_s9_equipo4.pdf
├── frontend/
│   └── index.html
├── models/
│   ├── dl_histories.npy
│   └── final_model.joblib          # Generado localmente
├── reports/
│   ├── figures/
│   │   ├── Figura1_eda_overview.png
│   │   ├── comparacion_modelos.png
│   │   ├── comparacion_homogenea.png
│   │   ├── confusion_matrix.png
│   │   ├── convergencia_dl.png
│   │   ├── efecto_balanceo.png
│   │   ├── shap_global.png
│   │   ├── shap_por_clase.png
│   │   └── ui_prediccion.png
│   ├── balancing_results.csv
│   ├── confusion_matrix.csv
│   ├── final_model_results.csv
│   ├── rf_refinement_results.csv
│   ├── homogeneous_comparison_results.csv
│   └── feature_redundancy_results.csv
├── src/
│   ├── balancing.py
│   ├── eda.py
│   ├── explain.py
│   ├── feature_redundancy_check.py
│   ├── final_model.py
│   ├── homogeneous_comparison.py
│   ├── preprocessing.py
│   ├── rf_refinement.py
│   ├── train_dl.py
│   └── train_ml.py
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-dl.txt
```

El archivo `models/final_model.joblib` no se publica en GitHub debido a su tamaño. Este archivo se genera localmente mediante `src/final_model.py`.

## Responsabilidad de cada script

| Archivo                          | Descripción                                                                                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `preprocessing.py`                | Limpieza, imputación, ingeniería de características y construcción de la variable objetivo                                               |
| `eda.py`                          | Análisis exploratorio reproducible y generación de la Figura 1                                                                            |
| `train_ml.py`                     | Comparación de Regresión Logística, Random Forest y XGBoost                                                                               |
| `train_dl.py`                     | Entrenamiento de tres arquitecturas MLP                                                                                                   |
| `balancing.py`                    | Comparación de datos sin balanceo, class weights, SMOTE y submuestreo (sobre Random Forest)                                               |
| `rf_refinement.py`                | Barrido de cantidad de árboles y profundidad del Random Forest                                                                            |
| `homogeneous_comparison.py`       | Reejecuta la comparación ML/DL aplicando SMOTE de forma homogénea a las cinco técnicas comparables                                        |
| `feature_redundancy_check.py`     | Compara el modelo final con y sin la variable `age`                                                                                       |
| `final_model.py`                  | Entrenamiento y evaluación final (sin `age`); genera precisión, recall, F1-score, accuracy, AUC-OVR macro, matriz de confusión y el modelo serializado |
| `explain.py`                      | Análisis SHAP global y local sobre el Random Forest final real (sin modelo sustituto)                                                     |
| `backend/app.py`                  | API FastAPI para predicción, validación, monitoreo y explicación local SHAP en cada respuesta                                             |
| `frontend/index.html`             | Interfaz web para ingresar datos, visualizar predicciones y consultar la explicación local de cada una                                    |

## Requisitos

### Entorno principal

- Windows, Linux o macOS.
- Python 3.12 recomendado.
- `pip` actualizado.
- Memoria suficiente para entrenar Random Forest, aplicar SMOTE y calcular SHAP.

Las dependencias principales están declaradas en `requirements.txt`.

### Deep Learning

Para ejecutar `src/train_dl.py` o `src/homogeneous_comparison.py` se requiere TensorFlow. Las dependencias adicionales están declaradas en `requirements-dl.txt`.

En Windows se recomienda utilizar Python 3.12 para mantener compatibilidad con TensorFlow.

## Instalación

### 1. Clonar el repositorio

```
git clone https://github.com/niconvc/acif104_s9_equipo4.git
cd acif104_s9_equipo4
```

### 2. Crear el entorno virtual

#### Windows PowerShell

```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea temporalmente la activación:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### Linux o macOS

```
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Actualizar pip

```
python -m pip install --upgrade pip
```

### 4. Instalar las dependencias principales

```
python -m pip install -r requirements.txt
```

### 5. Instalar las dependencias de Deep Learning

Este paso solo es necesario para ejecutar `src/train_dl.py` o `src/homogeneous_comparison.py`:

```
python -m pip install -r requirements-dl.txt
```

El archivo `requirements-dl.txt` incluye las dependencias principales y TensorFlow:

```
-r requirements.txt
tensorflow==2.21.0
```

## Ejecución del proyecto

Todos los comandos deben ejecutarse desde la carpeta raíz del repositorio:

```
acif104_s9_equipo4/
```

### 1. Verificar el preprocesamiento

```
python src/preprocessing.py
```

Salida esperada:

```
Filas finales: 50267

Distribución del target:
Económico    11746
Medio        29856
Premium       8665
```

### 2. Generar el análisis exploratorio

```
python src/eda.py
```

Archivo generado:

```
reports/figures/Figura1_eda_overview.png
```

### 3. Comparar modelos tradicionales

```
python src/train_ml.py
```

Modelos evaluados: Regresión Logística, Random Forest, XGBoost.

### 4. Comparar arquitecturas neuronales

Este paso requiere las dependencias de `requirements-dl.txt`:

```
python src/train_dl.py
```

Arquitecturas evaluadas: MLP Shallow, MLP Profunda, MLP con embeddings. Cada entrenamiento utiliza una instancia independiente de `EarlyStopping`, evitando que el callback conserve información de una red anterior.

### 5. Evaluar técnicas de balanceo

```
python src/balancing.py
```

Archivo generado: `reports/balancing_results.csv`. Técnicas evaluadas: sin balanceo, class weights, SMOTE, RandomUnderSampler (sobre Random Forest).

### 6. Ejecutar el refinamiento de Random Forest

```
python src/rf_refinement.py
```

Archivo generado: `reports/rf_refinement_results.csv`. Configuraciones evaluadas: 100, 200 y 300 árboles; profundidad máxima 20 y sin límite.

### 7. Validar la comparación homogénea de balanceo (opcional)

Requiere las dependencias de `requirements-dl.txt`:

```
python src/homogeneous_comparison.py
```

Archivo generado: `reports/homogeneous_comparison_results.csv`.

### 8. Validar la eliminación de la variable redundante (opcional)

```
python src/feature_redundancy_check.py
```

Archivo generado: `reports/feature_redundancy_results.csv`.

### 9. Entrenar y evaluar el modelo final

```
python src/final_model.py
```

Archivos generados:

```
models/final_model.joblib
reports/final_model_results.csv
reports/confusion_matrix.csv
reports/figures/confusion_matrix.png
```

Salida esperada de métricas globales:

```
Accuracy:       0.860
F1-macro:       0.849
AUC-OVR macro:  0.953
```

El modelo serializado utiliza compresión Joblib para reducir su tamaño, sin modificar sus predicciones ni métricas.

### 10. Generar el análisis SHAP

```
python src/explain.py
```

Archivos generados:

```
reports/figures/shap_global.png
reports/figures/shap_por_clase.png
```

El análisis se calcula directamente sobre el Random Forest final serializado en `models/final_model.joblib` (no sobre un modelo sustituto). Este paso puede tardar varios minutos.

## Ejecutar la aplicación web

Antes de iniciar la aplicación debe existir `models/final_model.joblib`. Si no existe, genéralo desde la raíz:

```
python src/final_model.py
```

Después, inicia FastAPI con Uvicorn:

```
python -m uvicorn backend.app:app
```

Para desarrollo con recarga automática:

```
python -m uvicorn backend.app:app --reload
```

Abre en el navegador:

```
http://127.0.0.1:8000
```

## Endpoints de la API

| Método | Ruta       | Descripción                                                                    |
| ------ | ---------- | ------------------------------------------------------------------------------- |
| `GET`  | `/`        | Muestra la interfaz web                                                        |
| `POST` | `/predict` | Predice el segmento y devuelve probabilidades estimadas y explicación local SHAP |
| `GET`  | `/metrics` | Devuelve métricas básicas de uso, distribución y latencia                       |
| `GET`  | `/health`  | Informa el estado del servicio y del modelo                                     |
| `GET`  | `/docs`    | Muestra la documentación interactiva de FastAPI                                 |

## Ejemplo de solicitud

```
{
  "model_year": 2019,
  "odometer": 25000,
  "cylinders": 8,
  "condition": "excellent",
  "fuel": "gas",
  "transmission": "automatic",
  "type": "truck",
  "paint_color": "black",
  "brand": "ford",
  "is_4wd": 1,
  "days_listed": 20
}
```

## Ejemplo de respuesta

```
{
  "segment": "Premium",
  "probabilities": {
    "Económico": 0.0,
    "Medio": 0.0,
    "Premium": 1.0
  },
  "probabilities_calibrated": false,
  "explanation": [
    {"variable": "Año del modelo", "shap_value": 0.2396, "direccion": "a favor"},
    {"variable": "Kilometraje", "shap_value": 0.1682, "direccion": "a favor"},
    {"variable": "Tracción 4×4", "shap_value": 0.0616, "direccion": "a favor"},
    {"variable": "Cantidad de cilindros", "shap_value": 0.0599, "direccion": "a favor"},
    {"variable": "type_truck", "shap_value": 0.0564, "direccion": "a favor"}
  ],
  "explanation_type": "shap_local",
  "latency_ms": 2580.0
}
```

Los valores de probabilidad y latencia pueden variar entre ejecuciones y equipos.

## Verificación del servicio

### Estado de la API

```
http://127.0.0.1:8000/health
```

Respuesta esperada:

```
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["Económico", "Medio", "Premium"],
  "probabilities_calibrated": false,
  "explanation_type": "shap_local"
}
```

### Monitoreo

```
http://127.0.0.1:8000/metrics
```

El endpoint `/metrics` registra durante la ejecución: número total de predicciones, distribución de predicciones por segmento, latencia media, latencia p95 y predicciones recientes. Estas métricas se mantienen en memoria y se reinician al detener o reiniciar el servidor.

## Documentación

La carpeta [`docs/`](docs/) contiene:

- [`acif104_s9_equipo4.pdf`](docs/acif104_s9_equipo4.pdf): informe de la fase intermedia del proyecto.

## Evidencias reproducibles

| Evidencia                                     | Archivo                                              |
| ---------------------------------------------- | ----------------------------------------------------- |
| EDA y distribución del target                  | `reports/figures/Figura1_eda_overview.png`            |
| Comparación de modelos (protocolo exploratorio)| `reports/figures/comparacion_modelos.png`             |
| Comparación de modelos (protocolo homogéneo)   | `reports/figures/comparacion_homogenea.png`           |
| Convergencia de redes neuronales               | `reports/figures/convergencia_dl.png`                 |
| Efecto del balanceo                            | `reports/figures/efecto_balanceo.png`                 |
| Barrido de Random Forest                       | `reports/rf_refinement_results.csv`                   |
| Comparación homogénea ML/DL                    | `reports/homogeneous_comparison_results.csv`          |
| Efecto de eliminar la variable `age`           | `reports/feature_redundancy_results.csv`              |
| Métricas finales, incluida AUC-OVR macro       | `reports/final_model_results.csv`                     |
| Matriz de confusión en CSV                     | `reports/confusion_matrix.csv`                        |
| Figura de matriz de confusión                  | `reports/figures/confusion_matrix.png`                |
| SHAP global (modelo final real)                | `reports/figures/shap_global.png`                     |
| SHAP por clase (modelo final real)             | `reports/figures/shap_por_clase.png`                  |
| Interfaz web                                   | `reports/figures/ui_prediccion.png`                   |
| Informe fase intermedia                        | `docs/acif104_s9_equipo4.pdf`                         |

## Consideraciones y limitaciones

- El dataset corresponde al mercado estadounidense y a publicaciones realizadas entre 2018 y 2019.
- Los umbrales de USD 5.000 y USD 20.000 son fijos y no se ajustan automáticamente por inflación.
- Las probabilidades del Random Forest no han sido calibradas. Un AUC elevado indica alta capacidad de discriminación, pero no garantiza probabilidades calibradas.
- La comparación de técnicas de balanceo con recall por clase (sección `balancing.py`) se realizó únicamente sobre Random Forest.
- Los tiempos de entrenamiento de las arquitecturas neuronales no fueron instrumentados de forma comparable con los modelos clásicos.
- El cálculo de SHAP local en tiempo de solicitud añade aproximadamente 2,6 segundos de latencia por predicción; para un servicio con mayor volumen de solicitudes concurrentes, esto debe optimizarse antes de un despliegue productivo real.
- Los resultados pueden variar ligeramente entre versiones de Python, sistemas operativos y bibliotecas.

## Archivos excluidos de Git

El archivo `.gitignore` excluye:

```
.venv/
venv/
__pycache__/
*.py[cod]
.idea/
*.iml
models/*.joblib
models/*.pkl
*.tmp
*.log
.DS_Store
Thumbs.db
Desktop.ini
```

El modelo final debe regenerarse localmente mediante:

```
python src/final_model.py
```

## Repositorio

<https://github.com/niconvc/acif104_s9_equipo4>
