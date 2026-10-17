"""
CARBOOST · Paso 3 — descarga de datos REALES de NASA POWER (opcional).

Ejecutar en un equipo con internet:
    python src/00_descargar_nasa_power.py

Descarga el registro diario 2015-2025 para Manaure (La Guajira) y lo guarda en
data/nasa_power_manaure.csv con los mismos nombres de columna que usa el proyecto.
La energía FV diaria se estima con el modelo físico del arreglo (60 kWp, PR 0.80,
coef. térmico -0.4 %/°C). Si la planta tiene contador real, reemplace la columna
e_fv_kwh por la medida del SCADA (eso es lo deseable).

Luego:
    python src/02_limpieza.py --fuente nasa
    python src/03_entrenar_evaluar.py
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LAT, LON = 11.78, -72.91
PARAMS = "ALLSKY_SFC_SW_DWN,TOA_SW_DWN,T2M,RH2M,WS2M,PRECTOTCORR"
URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters={PARAMS}&community=RE&longitude={LON}&latitude={LAT}"
    "&start=20150101&end=20251231&format=JSON"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "nasa_power_manaure.csv"


def main():
    print("Descargando", URL)
    with urllib.request.urlopen(URL, timeout=120) as r:
        js = json.load(r)
    p = js["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={
        "ALLSKY_SFC_SW_DWN": "ghi_kwh_m2", "TOA_SW_DWN": "toa_kwh_m2", "T2M": "t2m_c",
        "RH2M": "rh2m_pct", "WS2M": "ws2m_ms", "PRECTOTCORR": "lluvia_mm",
    })
    # Limpieza programada cada 14 días (se ajusta al plan real de la planta)
    df["dias_desde_limpieza"] = np.arange(len(df)) % 14
    tcell = df.t2m_c + 20 * (df.ghi_kwh_m2 / 6.5)
    pr = 0.80 * (1 - 0.004 * (tcell - 25))
    df["e_fv_kwh"] = (60 * df.ghi_kwh_m2 * pr).where(df.ghi_kwh_m2 > 0).round(1)
    df.index.name = "fecha"
    df.reset_index().assign(fecha=lambda d: d.fecha.dt.strftime("%Y-%m-%d")).to_csv(OUT, index=False)
    print("Guardado", OUT, len(df), "filas")


if __name__ == "__main__":
    main()
