# Clasificador de Segmento de Precio de Vehículos Usados

Proyecto de Aprendizaje de Máquinas — Fase 2. Clasifica vehículos usados
en tres segmentos de precio (**Económico**, **Medio**, **Premium**) a partir de
sus atributos, respondiendo a la pregunta *¿qué influye en la venta/precio de un auto?*

**Integrantes:** Gabriel Jara Rivas · Matías Lillo Yévenes · Nicolás Duarte Maldonado · Nicolás Navarrete Caro

## Resumen del modelo
- **Dataset:** `vehicles_us.csv` (~50.000 avisos, EE. UU.)
- **Problema:** clasificación multiclase (3 segmentos, desbalanceados 24/59/17 %)
- **Modelo final:** Random Forest + SMOTE — **F1-macro 0,85** en test (sin overfitting)
- **Explicabilidad:** SHAP (año, kilometraje y 4×4 son las variables más influyentes)

## Estructura del repositorio
```
car-price-segment/
├── data/          vehicles_us.csv
├── src/           preprocessing, train_ml, train_dl, balancing, final_model, explain
├── models/        final_model.joblib, dl_histories.npy
├── backend/       app.py (FastAPI)
├── frontend/      index.html
├── notebooks/     (vacío)
├── reports/       figures/, balancing_results.csv
├── requirements.txt
└── README.md
```

## Instalación
```bash
git clone https://github.com/[usuario]/car-price-segment.git
cd car-price-segment
python -m venv venv && source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Eso instala todo lo necesario para reproducir el modelo final y levantar el
sistema web. `train_dl.py` es lo único que además requiere TensorFlow (~250 MB):
```bash
pip install -r requirements-dl.txt   # incluye la base + tensorflow-cpu
```

## Ejecución

### 1. Reproducir el pipeline de modelado
Ejecutar **desde la raíz del proyecto**: los scripts leen `data/` y escriben en
`models/` con rutas relativas, así que fallan si se lanzan desde `src/`.
```bash
python src/preprocessing.py   # limpieza + feature engineering
python src/train_ml.py        # 3 modelos de Machine Learning
python src/train_dl.py        # 3 arquitecturas de Deep Learning
python src/balancing.py       # análisis de 3 técnicas de balanceo
python src/final_model.py     # entrena y guarda el modelo final (test)
python src/explain.py         # explicabilidad con SHAP
```
`train_dl.py` es el único que necesita `tensorflow-cpu`; el resto del proyecto,
incluido el sistema web, funciona sin él.

### 2. Levantar el sistema web (frontend + backend)
Requiere que `models/final_model.joblib` exista: el backend lo carga al importarse
y no arranca sin él. Si falta, generarlo antes con `python src/final_model.py`.
```bash
cd backend
uvicorn app:app --reload
```
O, sin cambiar de carpeta, desde la raíz:
```bash
uvicorn app:app --reload --app-dir backend
```
Luego abrir **http://127.0.0.1:8000** en el navegador.

## Endpoints de la API
| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Interfaz web |
| POST | `/predict` | Predice el segmento + probabilidades + explicación |
| GET | `/metrics` | Monitoreo: uso, latencia, distribución de clases |
| GET | `/health` | Estado del servicio |
