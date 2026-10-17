"""
CARBOOST · Paso 3 (Conseguir los datos) — generador de datos sintéticos justificados.

POR QUÉ SINTÉTICOS
  El entorno donde se construyó el proyecto no tenía acceso a la API de NASA POWER.
  Para no detener el proyecto se generó un registro diario SINTÉTICO, calibrado a la
  climatología de Manaure (La Guajira, 11.78 N, -72.91 O):
    * Irradiación global media mensual de 5.0 a 6.5 kWh/m2/día (orden de magnitud del
      Atlas de Radiación Solar IDEAM-UPME y de NASA POWER para la Alta Guajira).
    * Temporadas: seca y ventosa dic-jul (alisios), lluviosa sep-nov.
    * Irradiación extraterrestre (TOA) calculada con la geometría solar exacta.
  Si se dispone del CSV real (00_descargar_nasa_power.py), 02_limpieza.py lo usa en
  lugar de este archivo; el resto del flujo NO cambia.

QUÉ SE SIMULA
  Una desalinizadora de ósmosis inversa (RO) de agua de mar alimentada por un arreglo
  fotovoltaico (FV) de 60 kWp con batería. La variable objetivo es la ENERGÍA FV diaria
  entregada a la planta (kWh), medida por el contador del SCADA. Incluye pérdidas por
  temperatura, ensuciamiento por sal/polvo (se limpia cada 14 días o con lluvia > 4 mm)
  y fallas aleatorias del inversor.

  También se inyectan, a propósito, errores típicos de un registro real (huecos,
  centinelas -999, picos imposibles, ceros por reinicio de contador, filas duplicadas)
  para documentar la limpieza en el paso 4.

Semilla fija -> resultado reproducible.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(2026)
OUT = Path(__file__).resolve().parents[1] / "data" / "registro_planta_aquaboost_sintetico.csv"

LAT = 11.78
KWP = 60.0
PR_BASE = 0.80

# Climatología aproximada de GHI (kWh/m2/día) por mes, Alta Guajira
GHI_CLIM = np.array([5.6, 6.1, 6.5, 6.4, 6.0, 6.1, 6.4, 6.4, 6.0, 5.4, 5.0, 5.1])
# Índice de "humedad estacional" (0 seco ... 1 húmedo) por mes
WET = np.array([0.1, 0.0, 0.0, 0.3, 0.55, 0.3, 0.15, 0.2, 0.6, 1.0, 0.9, 0.4])
# Viento medio a 2 m (m/s) por mes: alisios fuertes en temporada seca
WIND = np.array([6.3, 6.6, 6.5, 6.0, 5.6, 6.3, 6.6, 5.8, 4.4, 3.6, 3.9, 5.4])


def toa_daily(doy, lat_deg=LAT):
    """Irradiación extraterrestre diaria sobre plano horizontal (kWh/m2/día)."""
    gsc = 1.367  # kW/m2
    phi = np.radians(lat_deg)
    b = 2 * np.pi * (doy - 1) / 365
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    h0 = (24 / np.pi) * gsc * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return h0


def smooth_monthly(values, doy):
    """Interpola cíclicamente un valor mensual a cada día del año."""
    mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    x = np.concatenate([mid - 365, mid, mid + 365])
    y = np.concatenate([values, values, values])
    return np.interp(doy, x, y)


def main():
    dates = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    h0 = toa_daily(doy)
    ghi_clim = smooth_monthly(GHI_CLIM, doy)
    kt_clim = ghi_clim / h0
    wet = smooth_monthly(WET, doy)
    wind_clim = smooth_monthly(WIND, doy)

    # Estado latente de humedad/nubosidad: AR(1) persistente
    m = np.zeros(n)
    for t in range(1, n):
        m[t] = 0.72 * m[t - 1] + np.sqrt(1 - 0.72 ** 2) * RNG.normal() + 0.25 * (wet[t] - 0.35)

    # Lluvia: más probable con humedad alta y en temporada húmeda
    p_rain = 1 / (1 + np.exp(-(-3.2 + 1.3 * m + 2.6 * wet)))
    rain = np.where(RNG.random(n) < p_rain, RNG.gamma(1.2, 6.0, n), 0.0)

    # Claridad del día t: depende del estado de hoy y de ayer (nubes que entran con la humedad)
    m_prev = np.concatenate([[0], m[:-1]])
    kt = kt_clim - 0.030 * m - 0.022 * m_prev - 0.07 * (rain > 2) * RNG.random(n) + 0.018 * RNG.normal(size=n)
    kt = np.clip(kt, 0.15, 0.76)
    ghi = kt * h0

    t2m = 27.6 + 0.9 * np.sin(2 * np.pi * (doy - 100) / 365) + 6 * (kt - kt_clim) + 0.5 * RNG.normal(size=n)
    rh2m = 70 + 7 * wet + 4.5 * m + 2.5 * RNG.normal(size=n)
    rh2m = np.clip(rh2m, 45, 98)
    ws2m = np.clip(wind_clim - 0.45 * m + 0.6 * RNG.normal(size=n), 0.5, 12)

    # Ensuciamiento (sal + polvo), limpieza programada cada 14 días o lluvia > 4 mm
    soil = np.zeros(n)
    dias_limpieza = np.zeros(n, dtype=int)
    s, d = 0.0, 0
    for t in range(n):
        if d >= 14:              # cuadrilla de limpieza (programada, conocida de antemano)
            s, d = 0.0, 0
        if rain[t] > 4:
            s *= 0.3
        dias_limpieza[t] = d
        soil[t] = s
        s = min(s + 0.0025 + 0.0004 * max(ws2m[t] - 5, 0), 0.12)
        d += 1

    # Rendimiento: temperatura de celda y ensuciamiento
    tcell = t2m + 20 * (ghi / 6.5)
    pr = PR_BASE * (1 - 0.004 * (tcell - 25)) * (1 - soil)
    # Fallas del inversor (eventos reales, NO son errores de dato)
    falla = RNG.random(n) < 0.010
    disp = np.where(falla, RNG.uniform(0.35, 0.75, n), 1.0)
    e_pv = KWP * ghi * pr * disp

    df = pd.DataFrame({
        "fecha": dates.strftime("%Y-%m-%d"),
        "ghi_kwh_m2": ghi.round(3),
        "toa_kwh_m2": h0.round(3),
        "t2m_c": t2m.round(2),
        "rh2m_pct": rh2m.round(1),
        "ws2m_ms": ws2m.round(2),
        "lluvia_mm": rain.round(1),
        "dias_desde_limpieza": dias_limpieza,
        "e_fv_kwh": e_pv.round(1),
    })

    # ---- Inyección de problemas de calidad típicos (se documentan en la limpieza) ----
    idx = lambda d: df.index[df.fecha == d][0]
    # 1) huecos del SCADA (comunicación caída)
    for d0, k in [("2016-03-10", 2), ("2018-07-21", 1), ("2019-10-02", 9), ("2023-05-14", 2), ("2024-08-30", 1)]:
        i = idx(d0)
        df.loc[i:i + k - 1, "e_fv_kwh"] = np.nan
    # 2) centinela -999 en variables meteorológicas (estilo NASA POWER)
    for col in ["ghi_kwh_m2", "rh2m_pct", "t2m_c", "ws2m_ms"]:
        ii = RNG.choice(n, 6, replace=False)
        df.loc[ii, col] = -999
    # 3) picos imposibles (registro duplicado del contador)
    for d0 in ["2017-02-11", "2020-06-03", "2021-11-19", "2024-03-07", "2025-01-22"]:
        df.loc[idx(d0), "e_fv_kwh"] = round(df.loc[idx(d0), "e_fv_kwh"] * 2.9, 1)
    # 4) ceros por reinicio del contador con sol normal
    for d0 in ["2016-12-05", "2019-04-17", "2022-09-08", "2025-06-12"]:
        df.loc[idx(d0), "e_fv_kwh"] = 0.0
    # 5) filas duplicadas (exportación repetida)
    dup = df.iloc[[400, 1500, 2600, 3300, 3900]]
    df = pd.concat([df, dup]).sort_values("fecha", kind="stable").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Escrito {OUT} con {len(df)} filas")
    print(df.describe().T[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
