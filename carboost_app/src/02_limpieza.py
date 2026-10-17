"""
CARBOOST · Paso 4 — Explorar y limpiar.

Cada regla de limpieza queda registrada en reports/bitacora_limpieza.json con
cuántas filas afectó y POR QUÉ se tomó la decisión.

Uso:  python src/02_limpieza.py            (datos sintéticos)
      python src/02_limpieza.py --fuente nasa
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KWP = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="sintetico", choices=["sintetico", "nasa"])
    a = ap.parse_args()
    src = ROOT / "data" / ("registro_planta_aquaboost_sintetico.csv" if a.fuente == "sintetico" else "nasa_power_manaure.csv")
    df = pd.read_csv(src)
    log = {"fuente": src.name, "filas_iniciales": int(len(df)), "reglas": []}

    def reg(regla, n, porque):
        log["reglas"].append({"regla": regla, "filas_afectadas": int(n), "por_que": porque})
        print(f"- {regla}: {n}")

    # R1 duplicados
    n = int(df.duplicated("fecha").sum())
    df = df.drop_duplicates("fecha", keep="first")
    reg("R1 Filas con fecha duplicada eliminadas", n,
        "Exportaciones repetidas del SCADA; contarlas dos veces sesga el entrenamiento.")

    df["fecha"] = pd.to_datetime(df.fecha)
    df = df.set_index("fecha").asfreq("D")   # fuerza calendario completo -> huecos visibles
    n_cal = int(df.e_fv_kwh.isna().sum())

    # R2 centinela -999
    met = ["ghi_kwh_m2", "t2m_c", "rh2m_pct", "ws2m_ms", "lluvia_mm"]
    n = int((df[met] <= -990).sum().sum())
    df[met] = df[met].mask(df[met] <= -990)
    reg("R2 Valores centinela -999 convertidos a faltante", n,
        "NASA POWER y muchos registradores usan -999 para 'sin dato'; usarlo como número destruye promedios.")

    # R3 límites físicos de GHI
    bad = (df.ghi_kwh_m2 <= 0) | (df.ghi_kwh_m2 > df.toa_kwh_m2)
    reg("R3 GHI fuera de (0, TOA] a faltante", bad.sum(),
        "La irradiación en superficie no puede superar la extraterrestre del mismo día.")
    df.loc[bad, "ghi_kwh_m2"] = np.nan

    # R4 picos imposibles de energía: PR implícito > 1
    pr_impl = df.e_fv_kwh / (KWP * df.ghi_kwh_m2)
    bad = pr_impl > 1.0
    reg("R4 Energía FV con rendimiento implícito > 1 (pico imposible) a faltante", bad.sum(),
        "Un arreglo de 60 kWp no puede entregar más energía que la que recibe; es doble registro del contador.")
    df.loc[bad, "e_fv_kwh"] = np.nan

    # R5 ceros con sol normal = reinicio del contador
    bad = (df.e_fv_kwh <= 1) & (df.ghi_kwh_m2 > 2)
    reg("R5 Energía = 0 con GHI > 2 kWh/m2 (reinicio de contador) a faltante", bad.sum(),
        "Si hubo sol y el contador marca cero, es un error de medición, no una falla real de la planta.")
    df.loc[bad, "e_fv_kwh"] = np.nan

    # R6 días con falla parcial del inversor: SE CONSERVAN
    pr_impl = df.e_fv_kwh / (KWP * df.ghi_kwh_m2)
    n_falla = int(((pr_impl < 0.55) & pr_impl.notna()).sum())
    reg("R6 Días con rendimiento bajo (falla parcial del inversor) CONSERVADOS", n_falla,
        "Son eventos reales de operación; borrarlos haría al modelo artificialmente optimista.")
    df["falla_sospechosa"] = (pr_impl < 0.55).astype(int)

    # R7 imputación meteorológica corta
    n = int(df[met].isna().sum().sum())
    df[met] = df[met].interpolate(limit=2, limit_area="inside")
    reg("R7 Huecos meteorológicos de 1-2 días interpolados linealmente", n - int(df[met].isna().sum().sum()),
        "Las variables de clima cambian suavemente de un día a otro; un hueco corto se interpola sin inventar información.")

    # R8 imputación física de energía en huecos <= 2 días
    na = df.e_fv_kwh.isna()
    grp = (na != na.shift()).cumsum()
    run = na.groupby(grp).transform("sum")
    pr_med = (df.e_fv_kwh / (KWP * df.ghi_kwh_m2)).rolling(7, min_periods=3).median().ffill()
    corto = na & (run <= 2) & df.ghi_kwh_m2.notna()
    df.loc[corto, "e_fv_kwh"] = (KWP * df.ghi_kwh_m2 * pr_med)[corto].round(1)
    df["e_imputada"] = corto.astype(int)
    reg("R8 Huecos de energía <= 2 días imputados con GHI x PR mediano de 7 días", corto.sum(),
        "Imputación física (no lineal): usa el sol medido ese día y el rendimiento reciente de la planta.")

    largo = df.e_fv_kwh.isna()
    reg("R9 Huecos largos (> 2 días) se dejan faltantes y no entran al entrenamiento", largo.sum(),
        "Rellenar 9 días seguidos sería inventar la respuesta; se excluyen esas filas.")

    log["filas_finales"] = int(len(df))
    log["faltantes_energia_tras_calendario"] = n_cal
    out = ROOT / "data" / "registro_limpio.csv"
    df.reset_index().to_csv(out, index=False)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "bitacora_limpieza.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print("Guardado", out)


if __name__ == "__main__":
    main()
