# Bank Marketing Predictor — Regresión Logística

Este proyecto implementa una **solución completa de inferencia** con Regresión Logística:

Datos → Preprocesamiento → Entrenamiento → Persistencia → API → Inferencia → Frontend

El caso es el dataset **Bank Marketing** de UCI: el modelo estima la probabilidad de que un cliente contrate un depósito a plazo a partir de información básica capturada por un asesor.[file:30]

---

## 1. Estructura del proyecto

```bash
bank-marketing-predictor/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI + rutas + frontend estático
│   ├── database.py         # SQLite para guardar inferencias
│   ├── model_service.py    # Carga del modelo y lógica de predicción
│   ├── schemas.py          # Modelos Pydantic (request/response)
│   └── static/
│       └── index.html      # Frontend mínimo (formulario web)
│
├── data/
│   └── bank.csv            # Dataset Bank Marketing (UCI)
│
├── models/
│   └── bank_marketing_pipeline.joblib  # Pipeline entrenado (scikit-learn)
│
├── scripts/
│   ├── train_model.py      # Entrenamiento + evaluación + persistencia
│   ├── run_backend.sh      # Levanta la API (opcional si usas bash)
│   ├── run_frontend.sh     # (opcional, no requerido si sirves estático)
│   └── run_site.sh         # (opcional)
│
├── requirements.txt
└── README.md
```

---

## 2. Requisitos

- Python 3.11 (recomendado).
- Git.
- Opcional: bash (Git Bash o WSL) si quieres usar los scripts `.sh` tal cual.[file:3]

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principales:

- `scikit-learn`, `pandas`, `joblib` para entrenamiento e inferencia.
- `fastapi`, `uvicorn`, `pydantic` para la API y validación de datos.[file:4][file:23]

---

## 3. Entrenamiento del modelo

Desde la carpeta `bank-marketing-predictor/`:

```bash
python scripts/train_model.py
```

Este script:

1. Carga `data/bank.csv` (dataset de UCI, separador `;`).[file:30]  
2. Selecciona las variables:

   - `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, `campaign`.
   - Variable objetivo: `y` (yes/no).  

3. Separa `X` e `y`.  
4. Divide en `train` y `test`.  
5. Construye un `ColumnTransformer` con:

   - Numéricas: `age`, `balance`, `campaign` → `StandardScaler`.  
   - Categóricas: `job`, `marital`, `education`, `housing`, `loan` → `OneHotEncoder`.  

6. Crea un `Pipeline` `preprocesamiento → LogisticRegression`.  
7. Entrena el modelo y calcula métricas (accuracy, precision, recall, F1).  
8. Guarda:

   - `models/bank_marketing_pipeline.joblib`
   - `models/metrics.json`.[file:29]

---

## 4. Levantar la API (FastAPI)

### Opción A — Comando directo

Desde `bank-marketing-predictor/`:

```bash
export PYTHONPATH=$(pwd)  # en PowerShell: $env:PYTHONPATH = (Get-Location).Path
uvicorn app.main:app --reload --host 0.0.0.0 --port 9001
```

La API quedará disponible en:

- Frontend: http://127.0.0.1:9001/
- Docs: http://127.0.0.1:9001/docs

### Opción B — Script `run_backend.sh`

Si tienes bash (Git Bash / WSL):

```bash
bash scripts/run_backend.sh
```

---

## 5. Endpoints disponibles

La aplicación expone:

- `GET /api/health`  
  - Devuelve `{ "status": "ok" }` para comprobar que la API está viva.[file:23]

- `POST /api/predict`  
  - Request (JSON):
    ```json
    {
      "age": 41,
      "job": "technician",
      "marital": "married",
      "education": "secondary",
      "balance": 3200,
      "housing": "yes",
      "loan": "no",
      "campaign": 2
    }
    ```
  - Response (JSON):
    ```json
    {
      "probability": 0.72,
      "prediction": "yes",
      "classification": "Potencialmente interesado"
    }
    ```

- `GET /api/predictions`  
  - Devuelve el historial reciente de inferencias guardadas en SQLite, incluyendo features de entrada, probabilidad, predicción y timestamp.[file:21][file:23]

---

## 6. Frontend (flujo Frontend → API → Modelo)

El archivo `app/static/index.html` implementa una interfaz mínima:

1. El usuario captura: edad, ocupación, estado civil, educación, balance, housing, loan, campaign.  
2. Al presionar “Estimar propensión”, el frontend hace un `fetch` a `POST /api/predict`.  
3. Muestra la probabilidad estimada y la clasificación en texto claro.  
4. Consulta `GET /api/predictions` para mostrar el historial reciente de inferencias.[file:23]

El frontend **no** entrena ni ejecuta el modelo directamente: solo consume la API y muestra el resultado.

---

# 7.2 Evidencias
Caso A — Mostrar una solicitud correcta y la predicción obtenida.
Caso B — Mostrar cómo responde la API ante datos inválidos.
Caso C — Captura de pantalla del frontend mostrando la probabilidad y la clasificación.

# 7.3 Respuestas teóricas (resumen corto)
¿Por qué el modelo se entrena fuera de /predict?
Porque el entrenamiento es costoso y se hace una sola vez; la API solo carga el pipeline ya entrenado para responder rápido y de forma estable. Mezclar entrenamiento con inferencia en el endpoint volvería el servicio lento y difícil de reproducir.

## ¿Por qué es clave usar el mismo preprocesamiento en inferencia y entrenamiento?
El modelo fue ajustado sobre datos escalados y codificados de cierta forma, si en producción transformas distinto (otras escalas, codificaciones distintas), las features dejan de representar lo mismo y las predicciones se vuelven poco confiables.

## Diferencia entre predict() y predict_proba() aquí
predict() devuelve solo la clase final, mientras que predict_proba() da la probabilidad numérica asociada a cada clase; usamos la probabilidad de la clase “yes” para interpretar la propensión a contratar el depósito.

## Si el modelo devuelve 0.72, ¿qué significa y qué NO significa?
Significa que, según el modelo entrenado con datos históricos, clientes con esas características tienen alrededor de 72% de probabilidad de pertenecer a la clase positiva (haber contratado el depósito); NO significa garantía de que ese cliente en particular vaya a contratarlo, ni una promesa de comportamiento futuro.

## ¿Por qué no usar duration?
duration es la duración de la llamada actual, solo se conoce después de llamar al cliente, así que no está disponible cuando quieres decidir a quién llamar. Usarlo como feature sería “ver el futuro” y el sistema no serviría para hacer la predicción antes del contacto.

## ¿Qué pasa si cambia la estructura de datos enviados por el frontend?
Si se renombra una variable o cambia el tipo, el esquema Pydantic fallará o el pipeline recibirá columnas erróneas; por eso versionamos el contrato de la API (CustomerFeatures), el modelo y el frontend para que siempre estén alineados