# CARBOOST Manaure

Misión de referencia: 1 · AquaBoost (Gradient Boosting · Desalinización RO + FV), Expedición ML Energética.

Aplicación web con Machine Learning (Gradient Boosting) para decidir, cada tarde, **cuánta agua comprometer para mañana** en una desalinizadora de ósmosis inversa alimentada con energía solar en Manaure, La Guajira.

Proyecto final · Machine Learning aplicado a sistemas energéticos · Universidad de La Guajira
Autor: Carlos Mario Gonzales Solano · 2026-II

## 1. Abrir la aplicación (sin instalar nada)

1. Descomprima el .zip.
2. Abra `web/index.html` con doble clic (Chrome, Edge o Firefox).

La aplicación funciona sin internet: el modelo entrenado está en `web/modelo.js` y se ejecuta en el navegador.
(Con internet solo se descargan las tipografías.)

Versión publicada: https://claude.ai/artifact/DKS1efqjzsyhnEVRgFk9p7

## 2. Reproducir el modelo (Python 3.10+)

```bash
pip install -r requirements.txt
python src/01_generar_datos_sinteticos.py   # Paso 3: datos (sintéticos calibrados)
python src/02_limpieza.py                   # Paso 4: limpieza documentada
python src/03_entrenar_evaluar.py           # Pasos 5-6: línea base, modelo, evaluación, exporta web/modelo.js
```

Tarda unos 2 minutos. Resultados en `reports/` (métricas, bitácora de limpieza y figuras).

### Con datos reales de NASA POWER

```bash
python src/00_descargar_nasa_power.py
python src/02_limpieza.py --fuente nasa
python src/03_entrenar_evaluar.py
```

## 3. Estructura

```
src/00_descargar_nasa_power.py      descarga de datos reales (opcional)
src/01_generar_datos_sinteticos.py  registro diario 2015-2025 calibrado a Manaure
src/02_limpieza.py                  9 reglas de limpieza -> reports/bitacora_limpieza.json
src/03_entrenar_evaluar.py          persistencia, climatología, Gradient Boosting cuantil, costo en COP
data/                               datos crudos y limpios
reports/                            métricas y figuras del informe
web/index.html                      aplicación web
web/modelo.js                       9 modelos (P10..P90) exportados de scikit-learn
web/ejemplo_ultimos_10_dias.csv     archivo de ejemplo para el modo "Archivo CSV"
```

## 4. Resultado principal (periodo de prueba oct. 2023 - dic. 2025)

| Método | MAE kWh | Costo COP/año |
|---|---|---|
| Persistencia (mañana = hoy) | 13,6 | 13 262 341 |
| Climatología mensual | 17,3 | 15 659 715 |
| Gradient Boosting P50 | 11,6 | 10 641 242 |
| **Gradient Boosting, plan en q* = 0,70** | 13,1 | **9 575 036** |

Límite: los datos son sintéticos calibrados; antes de usarlo en una planta real hay que reentrenar con su registro SCADA y con NASA POWER.
