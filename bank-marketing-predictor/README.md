# 7.1 Cómo correr
Crear entorno, instalar requirements.

Ejecutar python scripts/train_model.py.

Ejecutar bash scripts/run_backend.sh.

Abrir http://127.0.0.1:9001 en el navegador.

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