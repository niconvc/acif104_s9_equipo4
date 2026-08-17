# Clasificador de Segmento de Precio de Vehículos Usados

Proyecto de Aprendizaje de Máquinas correspondiente a la tercera fase del proyecto. La solución clasifica vehículos usados en tres segmentos de precio —**Económico**, **Medio** y **Premium**— utilizando sus características técnicas y comerciales.

La pregunta que orienta el proyecto es:

> ¿Qué características de un vehículo usado permiten clasificarlo en un segmento de precio y qué estrategia de modelado ofrece el mejor equilibrio de desempeño entre las tres clases?

## Integrantes

- Gabriel Jara Rivas
- Matías Lillo Yévenes
- Nicolás Duarte Maldonado
- Nicolás Navarrete Caro

## Descripción del problema

A partir del dataset `vehicles_us.csv`, el proyecto formula una tarea de clasificación multiclase. La variable objetivo se construye utilizando los siguientes umbrales:

| Segmento | Definición |
|---|---|
| Económico | Precio inferior a USD 5.000 |
| Medio | Precio entre USD 5.000 y USD 20.000 |
| Premium | Precio superior a USD 20.000 |

Después del proceso de limpieza, el conjunto utilizado para el modelado contiene **50.267 vehículos**, distribuidos de la siguiente manera:

| Segmento | Cantidad | Porcentaje |
|---|---:|---:|
| Económico | 11.746 | 23,4 % |
| Medio | 29.856 | 59,4 % |
| Premium | 8.665 | 17,2 % |
| **Total** | **50.267** | **100,0 %** |

Esta distribución presenta un desbalance relevante, principalmente por el predominio de la clase Medio. Por esta razón, la métrica principal del proyecto es el **F1-macro**, complementada con accuracy, precisión y recall por clase.

## Resultado del modelo final

El modelo seleccionado es un pipeline compuesto por:

1. Preprocesamiento numérico y categórico.
2. Balanceo mediante SMOTE.
3. Clasificación mediante Random Forest de 200 árboles.

El modelo se entrenó con los conjuntos de entrenamiento y validación y se evaluó sobre **7.541 vehículos de test no utilizados durante el entrenamiento**.

| Segmento | Precisión | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| Económico | 0,796 | 0,780 | 0,788 | 1.762 |
| Medio | 0,879 | 0,885 | 0,882 | 4.479 |
| Premium | 0,868 | 0,872 | 0,870 | 1.300 |
| Global (macro) | 0,847 | 0,845 | 0,846 | — |

Resultados globales:

- **Accuracy:** 0,858.
- **F1-macro:** 0,846.
- **Casos de test:** 7.541.
- **Tamaño comprimido del modelo:** aproximadamente 63,1 MB.

Las métricas internas corresponden al dataset analizado y no garantizan el mismo desempeño en otros mercados, periodos o fuentes de datos.

## Explicabilidad y alcance

El análisis de explicabilidad global utiliza SHAP sobre un Random Forest sustituto de 120 árboles y profundidad máxima 14. Este modelo compacto permite reducir el costo computacional del análisis, pero no corresponde exactamente al modelo final utilizado por la API.

Las variables globalmente más relevantes incluyen:

- Año del modelo.
- Antigüedad.
- Odómetro.
- Tracción 4×4.
- Cantidad de cilindros.
- Tipo de carrocería.

`model_year` y `age` contienen información redundante, debido a que:

```text
age = 2019 - model_year
```

La aplicación web no calcula SHAP por instancia. La interfaz presenta una **orientación heurística** basada en variables relevantes identificadas mediante el análisis exploratorio y SHAP global.

Las probabilidades entregadas por el clasificador no han sido sometidas a un procedimiento específico de calibración, por lo que deben considerarse estimaciones orientativas y no niveles de confianza perfectamente calibrados.

## Estructura del repositorio

```text
car-price-segment/
├── backend/
│   └── app.py
├── data/
│   └── vehicles_us.csv
├── docs/
│   ├── acif104_s9_grupo4.pdf
│   └── demostracion_sistema_funcional.mp4
├── frontend/
│   └── index.html
├── models/
│   └── final_model.joblib
├── reports/
│   ├── figures/
│   │   ├── Figura1_eda_overview.png
│   │   ├── confusion_matrix.png
│   │   ├── shap_global.png
│   │   └── shap_por_clase.png
│   ├── balancing_results.csv
│   ├── confusion_matrix.csv
│   ├── final_model_results.csv
│   └── rf_refinement_results.csv
├── src/
│   ├── balancing.py
│   ├── eda.py
│   ├── explain.py
│   ├── final_model.py
│   ├── preprocessing.py
│   ├── rf_refinement.py
│   ├── train_dl.py
│   └── train_ml.py
├── .gitignore
├── requirements.txt
├── requirements-dl.txt
└── README.md
```

El archivo `models/final_model.joblib` no se publica en GitHub debido a su tamaño. Se genera localmente mediante `src/final_model.py`.

## Responsabilidad de cada script

| Archivo | Descripción |
|---|---|
| `preprocessing.py` | Limpieza, imputación, ingeniería de características y construcción del target |
| `train_ml.py` | Comparación de Regresión Logística, Random Forest y XGBoost |
| `train_dl.py` | Entrenamiento de tres arquitecturas MLP |
| `balancing.py` | Comparación de datos sin balanceo, class weights, SMOTE y submuestreo |
| `rf_refinement.py` | Barrido de cantidad de árboles y profundidad del Random Forest |
| `final_model.py` | Entrenamiento, evaluación, serialización y generación de evidencias finales |
| `explain.py` | Análisis SHAP global sobre un Random Forest sustituto |
| `eda.py` | Generación reproducible de la Figura 1 del análisis exploratorio |

## Requisitos

### Entorno principal

- Windows, Linux o macOS.
- Python 3.12 recomendado.
- `pip` actualizado.
- Memoria suficiente para entrenar Random Forest y aplicar SMOTE.

Las dependencias principales están declaradas en:

```text
requirements.txt
```

### Deep Learning

Para ejecutar `src/train_dl.py` se requiere TensorFlow. Las dependencias adicionales están declaradas en:

```text
requirements-dl.txt
```

En Windows se recomienda Python 3.12 para mantener compatibilidad con TensorFlow.

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/niconvc/car-price-segment.git
cd car-price-segment
```

### 2. Crear el entorno virtual

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea temporalmente la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar las dependencias principales

```bash
python -m pip install -r requirements.txt
```

### 5. Instalar las dependencias de Deep Learning

Este paso solo es necesario para ejecutar `train_dl.py`:

```bash
python -m pip install -r requirements-dl.txt
```

`requirements-dl.txt` incluye las dependencias principales y TensorFlow:

```text
-r requirements.txt
tensorflow==2.21.0
```

## Ejecución del proyecto

Todos los comandos deben ejecutarse desde la carpeta raíz del repositorio:

```text
car-price-segment/
```

### 1. Verificar el preprocesamiento

```bash
python src/preprocessing.py
```

Salida esperada:

```text
Filas finales: 50267

Distribución del target:
Económico    11746
Medio        29856
Premium       8665
```

### 2. Generar el análisis exploratorio

```bash
python src/eda.py
```

Archivo generado:

```text
reports/figures/Figura1_eda_overview.png
```

La figura utiliza:

- Verde para Económico.
- Naranja para Medio.
- Rojo para Premium.

### 3. Comparar modelos tradicionales

```bash
python src/train_ml.py
```

Modelos evaluados:

- Regresión Logística.
- Random Forest.
- XGBoost.

### 4. Comparar arquitecturas neuronales

Requiere las dependencias de `requirements-dl.txt`:

```bash
python src/train_dl.py
```

Arquitecturas evaluadas:

- MLP Shallow.
- MLP Profunda.
- MLP con embeddings.

Cada entrenamiento utiliza una instancia independiente de `EarlyStopping`.

### 5. Evaluar técnicas de balanceo

```bash
python src/balancing.py
```

Archivo generado:

```text
reports/balancing_results.csv
```

Técnicas evaluadas:

- Sin balanceo.
- Class weights.
- SMOTE.
- RandomUnderSampler.

### 6. Ejecutar el refinamiento de Random Forest

```bash
python src/rf_refinement.py
```

Archivo generado:

```text
reports/rf_refinement_results.csv
```

Configuraciones evaluadas:

- 100, 200 y 300 árboles.
- Profundidad máxima 20.
- Profundidad sin límite.

El script registra:

- F1-macro.
- Recall Premium.
- Recall Económico.
- Tiempo de entrenamiento.
- Tamaño serializado.

### 7. Entrenar y evaluar el modelo final

```bash
python src/final_model.py
```

Archivos generados:

```text
models/final_model.joblib
reports/final_model_results.csv
reports/confusion_matrix.csv
reports/figures/confusion_matrix.png
```

El modelo serializado utiliza compresión Joblib para reducir el tamaño del archivo.

### 8. Generar el análisis SHAP

```bash
python src/explain.py
```

Archivos generados:

```text
reports/figures/shap_global.png
reports/figures/shap_por_clase.png
```

El análisis utiliza un Random Forest sustituto de 120 árboles y profundidad máxima 14.

## Ejecutar la aplicación web

Antes de iniciar la aplicación debe existir:

```text
models/final_model.joblib
```

Si el archivo no existe, ejecútalo desde la raíz:

```bash
python src/final_model.py
```

Después inicia FastAPI con Uvicorn:

```bash
python -m uvicorn backend.app:app
```

Para desarrollo con recarga automática:

```bash
python -m uvicorn backend.app:app --reload
```

Abre en el navegador:

```text
http://127.0.0.1:8000
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Muestra la interfaz web. |
| `POST` | `/predict` | Predice el segmento y devuelve las probabilidades estimadas y la orientación heurística. |
| `GET` | `/metrics` | Devuelve las métricas básicas de uso y latencia. |
| `GET` | `/health` | Informa el estado del servicio y del modelo. |
| `GET` | `/docs` | Muestra la documentación interactiva de FastAPI. |

## Ejemplo de solicitud

```json
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

```json
{
  "segment": "Premium",
  "probabilities": {
    "Económico": 0.0,
    "Medio": 0.02,
    "Premium": 0.98
  },
  "probabilities_calibrated": false,
  "explanation": [
    "Año reciente: atributo generalmente asociado con segmentos de mayor precio.",
    "Kilometraje bajo: factor habitualmente asociado con un mayor valor.",
    "Tracción 4×4: característica asociada con un mayor valor en el conjunto analizado."
  ],
  "explanation_type": "heuristic",
  "latency_ms": 53.0
}
```

Los valores de probabilidad y latencia pueden variar entre ejecuciones y equipos.

## Verificación del servicio

Estado de la API:

```text
http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": [
    "Económico",
    "Medio",
    "Premium"
  ],
  "probabilities_calibrated": false,
  "explanation_type": "heuristic"
}
```

Monitoreo:

```text
http://127.0.0.1:8000/metrics
```
El endpoint `/metrics` registra durante la ejecución:

- Número total de predicciones.
- Distribución de predicciones por segmento.
- Latencia media.
- Latencia p95.
- Predicciones recientes.

Estas métricas se mantienen en memoria y se reinician al detener o reiniciar el servidor.

## Documentación y demostración

La carpeta [`docs/`](docs/) contiene los siguientes archivos:

- [`acif104_s9_grupo4.pdf`](docs/acif104_s9_grupo4.pdf): informe final de la Sumativa 2, con la problemática, metodología, desarrollo, evaluación, limitaciones, propuestas de mejora y conclusiones del proyecto.
- [`demostracion_sistema_funcional.mp4`](docs/demostracion_sistema_funcional.mp4): evidencia audiovisual de la operación del sistema.

El video presenta:

- El funcionamiento general de la interfaz.
- Predicciones para los segmentos Económico, Medio y Premium.
- Las probabilidades estimadas para cada clase.
- Las orientaciones heurísticas.
- La actualización del panel de monitoreo.
- La validación de datos fuera de rango.

## Evidencias reproducibles

| Evidencia | Archivo |
|---|---|
| EDA y distribución del target | `reports/figures/Figura1_eda_overview.png` |
| Comparación de modelos | `reports/figures/comparacion_modelos.png` |
| Convergencia de redes neuronales | `reports/figures/convergencia_dl.png` |
| Efecto del balanceo | `reports/figures/efecto_balanceo.png` |
| Barrido de Random Forest | `reports/rf_refinement_results.csv` |
| Resultados del modelo final | `reports/final_model_results.csv` |
| Matriz de confusión | `reports/confusion_matrix.csv` |
| Figura de matriz de confusión | `reports/figures/confusion_matrix.png` |
| SHAP global | `reports/figures/shap_global.png` |
| SHAP por clase | `reports/figures/shap_por_clase.png` |
| Interfaz web | `reports/figures/ui_prediccion.png` |

## Consideraciones

- El dataset corresponde al mercado estadounidense y a un periodo comprendido entre 2018 y 2019.
- Los umbrales de USD 5.000 y USD 20.000 son fijos y no se ajustan automáticamente por inflación.
- Las probabilidades del Random Forest no han sido calibradas.
- El análisis SHAP utiliza un modelo sustituto y no exactamente el pipeline utilizado por la API.
- La interfaz entrega reglas descriptivas, no atribuciones SHAP locales.
- `model_year` y `age` contienen información redundante.
- Los resultados pueden variar ligeramente entre versiones de Python y de las bibliotecas.

## Archivos excluidos de Git

El archivo `.gitignore` excluye:

```text
.venv/
__pycache__/
*.py[cod]
.idea/
models/*.joblib
models/*.pkl
```

El modelo final debe regenerarse localmente mediante:

```bash
python src/final_model.py
```

## Repositorio

[https://github.com/niconvc/car-price-segment](https://github.com/niconvc/car-price-segment)