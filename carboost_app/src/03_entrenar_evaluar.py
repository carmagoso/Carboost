"""
CARBOOST · Pasos 5 y 6 — Línea base, modelo y evaluación con datos nuevos.

Pregunta: a las 18:00 de hoy, ¿cuánta energía FV tendrá MAÑANA la desalinizadora?
Decisión: cuántos m3 de agua comprometer (y cuántas horas de RO programar) mañana.

  * Partición CRONOLÓGICA 60/20/20 (entrenamiento / validación / prueba). Nunca aleatoria.
  * Líneas base: persistencia ("mañana = hoy") y climatología mensual.
  * Modelo: Gradient Boosting (scikit-learn) con pérdida cuantil -> P10 ... P90.
  * Métrica de negocio: pesos colombianos (COP) por error de planeación.
  * Exporta el modelo a JSON para que la app web lo ejecute en el navegador.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)
WEB = ROOT / "web"

# ---- Parámetros de la planta y del negocio (supuestos declarados) ----
SEC = 4.0           # kWh por m3 de agua producto (SWRO incluida la captación)
RO_M3H = 5.0        # capacidad de la RO, m3/h
DIESEL_COP_KWH = 1600   # costo de cubrir 1 kWh faltante con el grupo electrógeno
AGUA_COP_M3 = 15000     # valor de 1 m3 de agua que no se produjo (reemplazo por carrotanque)
C_SOBRE = DIESEL_COP_KWH            # por kWh planeado de más
C_BAJO = AGUA_COP_M3 / SEC          # por kWh solar desperdiciado (agua no producida)
Q_STAR = C_BAJO / (C_BAJO + C_SOBRE)
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FEATURES = [
    "e_hoy", "e_ayer", "e_anteayer", "e_media7", "kt_hoy", "kt_media3",
    "rh_hoy", "rh_delta", "viento_hoy", "temp_hoy", "lluvia_hoy", "lluvia_3d",
    "dias_limpieza_man", "dias_sin_lluvia", "toa_man", "doy_sin", "doy_cos",
]
NOMBRES = {
    "e_hoy": "Energía FV hoy", "e_ayer": "Energía FV ayer", "e_anteayer": "Energía FV anteayer",
    "e_media7": "Media 7 días energía", "kt_hoy": "Índice de claridad hoy", "kt_media3": "Claridad media 3 días",
    "rh_hoy": "Humedad relativa hoy", "rh_delta": "Cambio de humedad (hoy-ayer)", "viento_hoy": "Viento hoy",
    "temp_hoy": "Temperatura hoy", "lluvia_hoy": "Lluvia hoy", "lluvia_3d": "Lluvia 3 días",
    "dias_limpieza_man": "Días desde limpieza (mañana)", "dias_sin_lluvia": "Días sin lluvia > 4 mm",
    "toa_man": "Irradiación extraterrestre mañana", "doy_sin": "Estacionalidad (seno)", "doy_cos": "Estacionalidad (coseno)",
}


def costo(plan, real):
    """Costo en COP de planear `plan` kWh cuando la planta tuvo `real` kWh."""
    return C_SOBRE * np.maximum(plan - real, 0) + C_BAJO * np.maximum(real - plan, 0)


def construir_features(df):
    d = pd.DataFrame(index=df.index)
    e = df.e_fv_kwh
    d["e_hoy"] = e
    d["e_ayer"] = e.shift(1)
    d["e_anteayer"] = e.shift(2)
    d["e_media7"] = e.rolling(7, min_periods=7).mean()
    kt = df.ghi_kwh_m2 / df.toa_kwh_m2
    d["kt_hoy"] = kt
    d["kt_media3"] = kt.rolling(3).mean()
    d["rh_hoy"] = df.rh2m_pct
    d["rh_delta"] = df.rh2m_pct.diff()
    d["viento_hoy"] = df.ws2m_ms
    d["temp_hoy"] = df.t2m_c
    d["lluvia_hoy"] = df.lluvia_mm
    d["lluvia_3d"] = df.lluvia_mm.rolling(3).sum()
    d["dias_limpieza_man"] = df.dias_desde_limpieza.shift(-1)       # el plan de limpieza se conoce de antemano
    lluvia_fuerte = (df.lluvia_mm > 4).astype(int)
    d["dias_sin_lluvia"] = lluvia_fuerte.groupby(lluvia_fuerte.cumsum()).cumcount()
    d["toa_man"] = df.toa_kwh_m2.shift(-1)                           # astronómico: conocido de antemano
    doy_m = (df.index + pd.Timedelta(days=1)).dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy_m / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy_m / 365.25)
    d["y_manana"] = e.shift(-1)
    d["clim_mes"] = np.nan
    return d


def exportar_gbr(m):
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({
            "f": t.feature.tolist(),
            "t": [round(float(x), 5) for x in t.threshold],
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": [round(float(x), 4) for x in t.value[:, 0, 0]],
        })
    return {"init": float(m.init_.constant_.ravel()[0]), "lr": m.learning_rate, "trees": trees}


def main():
    df = pd.read_csv(ROOT / "data" / "registro_limpio.csv", parse_dates=["fecha"]).set_index("fecha")
    d = construir_features(df)
    d = d.dropna(subset=FEATURES + ["y_manana"])
    n = len(d)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    print(f"Filas útiles {n}: train {tr.index[0].date()}..{tr.index[-1].date()} ({len(tr)}), "
          f"val ..{va.index[-1].date()} ({len(va)}), test {te.index[0].date()}..{te.index[-1].date()} ({len(te)})")

    # Climatología mensual aprendida SOLO del entrenamiento
    clim = tr.groupby((tr.index + pd.Timedelta(days=1)).month).y_manana.mean()
    for part in (tr, va, te, d):
        part["clim_mes"] = (part.index + pd.Timedelta(days=1)).month.map(clim).values

    # ---- Selección de hiperparámetros en VALIDACIÓN (P50, MAE) ----
    grid = []
    for n_est in [150, 250, 400]:
        for depth in [2, 3, 4]:
            for lr in [0.03, 0.06]:
                m = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=n_est, max_depth=depth,
                                              learning_rate=lr, subsample=0.8, min_samples_leaf=20, random_state=0)
                m.fit(tr[FEATURES], tr.y_manana)
                mae = mean_absolute_error(va.y_manana, m.predict(va[FEATURES]))
                grid.append({"n_estimators": n_est, "max_depth": depth, "learning_rate": lr, "mae_val": round(mae, 2)})
    grid = sorted(grid, key=lambda g: g["mae_val"])
    best = {k: grid[0][k] for k in ("n_estimators", "max_depth", "learning_rate")}
    print("Mejor configuración (validación):", grid[0])

    # Validación: también la persistencia, para mostrar honestamente
    val_res = {
        "persistencia_mae": mean_absolute_error(va.y_manana, va.e_hoy),
        "climatologia_mae": mean_absolute_error(va.y_manana, va.clim_mes),
        "gb_p50_mae": grid[0]["mae_val"],
    }

    # ---- Reentrenar con train+val y evaluar en PRUEBA (nunca vista) ----
    trva = pd.concat([tr, va])
    modelos = {}
    t0 = time.time()
    for q in QUANTILES:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, subsample=0.8, min_samples_leaf=20,
                                      random_state=0, **best)
        m.fit(trva[FEATURES], trva.y_manana)
        modelos[q] = m
    print(f"Entrenados {len(QUANTILES)} cuantiles en {time.time()-t0:.1f}s")

    P = np.column_stack([modelos[q].predict(te[FEATURES]) for q in QUANTILES])
    P = np.sort(P, axis=1)   # evita cruces de cuantiles
    y = te.y_manana.values

    def q_interp(Prow, q):
        return np.interp(q, QUANTILES, Prow)
    plan_gb = np.array([q_interp(r, Q_STAR) for r in P])
    p50 = P[:, QUANTILES.index(0.5)]
    p10, p90 = P[:, 0], P[:, -1]

    res = {}
    for nombre, pred in [("Persistencia (mañana = hoy)", te.e_hoy.values),
                         ("Climatología mensual", te.clim_mes.values),
                         ("Gradient Boosting P50", p50),
                         (f"Gradient Boosting cuantil óptimo q*={Q_STAR:.2f}", plan_gb),
                         ("Oráculo (conocer el futuro)", y)]:
        c = costo(pred, y)
        res[nombre] = {
            "mae_kwh": round(float(mean_absolute_error(y, pred)), 2),
            "rmse_kwh": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            "mae_m3": round(float(mean_absolute_error(y, pred)) / SEC, 2),
            "costo_total_cop": round(float(c.sum())),
            "costo_anual_cop": round(float(c.sum() / len(y) * 365)),
            "diesel_kwh_anual": round(float(np.maximum(pred - y, 0).sum() / len(y) * 365)),
            "agua_perdida_m3_anual": round(float(np.maximum(y - pred, 0).sum() / SEC / len(y) * 365)),
        }
    cobertura = float(np.mean((y >= p10) & (y <= p90)))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("Cobertura P10-P90 en prueba:", round(cobertura, 3))

    base = res["Persistencia (mañana = hoy)"]["costo_anual_cop"]
    modelo = res[f"Gradient Boosting cuantil óptimo q*={Q_STAR:.2f}"]["costo_anual_cop"]
    ahorro = base - modelo

    # Robustez: costo por año calendario de prueba
    by_year = {}
    yrs = te.index.year
    for yr in sorted(set(yrs)):
        mk = yrs == yr
        by_year[int(yr)] = {"persistencia": round(float(costo(te.e_hoy.values[mk], y[mk]).sum())),
                            "gb": round(float(costo(plan_gb[mk], y[mk]).sum())), "dias": int(mk.sum())}

    # Errores grandes: ¿cuándo falla el modelo?
    err = np.abs(p50 - y)
    peor = te.assign(p50=p50, err=err).sort_values("err", ascending=False).head(8)
    peores = [{"fecha_decision": str(i.date()), "real": round(r.y_manana, 1), "p50": round(r.p50, 1),
               "lluvia_man": None} for i, r in peor.iterrows()]

    # Importancia de variables (P50)
    imp = pd.Series(modelos[0.5].feature_importances_, index=FEATURES).sort_values(ascending=False)

    # ---------------- Figuras ----------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    C1, C2, C3, C4 = "#0e7490", "#d97706", "#64748b", "#7c3aed"

    raw = pd.read_csv(ROOT / "data" / "registro_planta_aquaboost_sintetico.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    s = df.e_fv_kwh
    ax.plot(s.index, s.values, lw=0.4, color=C1)
    ax.plot(s.rolling(30, center=True).mean(), lw=1.4, color="#0f172a", label="media móvil 30 días")
    for a, b, lab, col in [(tr.index[0], tr.index[-1], "Entrenamiento 60 %", "#e0f2fe"),
                           (va.index[0], va.index[-1], "Validación 20 %", "#fef3c7"),
                           (te.index[0], te.index[-1], "Prueba 20 %", "#ede9fe")]:
        ax.axvspan(a, b, color=col, zorder=0, label=lab)
    ax.set_ylabel("kWh/día"); ax.set_title("Energía FV diaria de la planta y partición cronológica", loc="left")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "f1_serie_particion.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    mes = df.index.month
    axs[0].boxplot([df.e_fv_kwh[mes == k].dropna() for k in range(1, 13)], showfliers=True,
                   flierprops={"markersize": 2})
    axs[0].set_xticks(range(1, 13), list("EFMAMJJASOND")); axs[0].set_ylabel("kWh/día")
    axs[0].set_title("Estacionalidad: oct-nov es la época difícil", loc="left")
    rr = raw[(raw.ghi_kwh_m2 > 0)]
    axs[1].scatter(rr.ghi_kwh_m2, rr.e_fv_kwh, s=2, color=C3, alpha=0.4, label="registro crudo")
    flag = rr[(rr.e_fv_kwh / (60 * rr.ghi_kwh_m2) > 1) | ((rr.e_fv_kwh <= 1) & (rr.ghi_kwh_m2 > 2))]
    axs[1].scatter(flag.ghi_kwh_m2, flag.e_fv_kwh, s=18, color="#dc2626", label="errores (R4, R5)")
    fl = df[df.falla_sospechosa == 1]
    axs[1].scatter(fl.ghi_kwh_m2, fl.e_fv_kwh, s=10, facecolors="none", edgecolors=C2, label="fallas reales (R6)")
    axs[1].set_xlabel("GHI kWh/m²/día"); axs[1].set_ylabel("kWh/día")
    axs[1].set_title("Detección de atípicos", loc="left"); axs[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "f2_exploracion.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = slice(-120, None)
    xi = te.index[w] + pd.Timedelta(days=1)
    ax.fill_between(xi, p10[w], p90[w], color=C1, alpha=0.18, label="banda P10–P90")
    ax.plot(xi, y[w], color="#0f172a", lw=1.2, label="real")
    ax.plot(xi, p50[w], color=C1, lw=1.2, label="GB P50")
    ax.plot(xi, te.e_hoy.values[w], color=C2, lw=0.9, ls="--", label="persistencia")
    ax.set_ylabel("kWh/día"); ax.set_title("Prueba (últimos 120 días): pronóstico vs real", loc="left")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "f3_pronostico_prueba.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    nombres = ["Persistencia", "Climatología", "GB P50", f"GB q*={Q_STAR:.2f}"]
    keys = list(res.keys())[:4]
    vals = [res[k]["costo_anual_cop"] / 1e6 for k in keys]
    bars = axs[0].barh(nombres, vals, color=[C2, C3, C1, "#155e75"])
    for b, v in zip(bars, vals):
        axs[0].text(v, b.get_y() + b.get_height() / 2, f" {v:.1f}", va="center", fontsize=8)
    axs[0].invert_yaxis(); axs[0].set_xlabel("Millones COP / año"); axs[0].set_title("Costo de planear mal (prueba)", loc="left")
    top = imp.head(8)[::-1]
    axs[1].barh([NOMBRES[k] for k in top.index], top.values, color=C1)
    axs[1].set_title("Importancia de variables (P50)", loc="left"); axs[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(FIG / "f4_costo_importancia.png", dpi=200); plt.close(fig)

    # Curva de costo vs cuantil elegido (validación de la regla q*)
    qs = np.linspace(0.1, 0.9, 33)
    cq = [costo(np.array([q_interp(r, q) for r in P]), y).sum() / len(y) * 365 / 1e6 for q in qs]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(qs, cq, color=C1); ax.axvline(Q_STAR, color=C2, ls="--", lw=1)
    ax.text(Q_STAR, max(cq), f" q* = {Q_STAR:.2f}", color=C2, va="top", fontsize=8)
    ax.set_xlabel("Cuantil usado para planear"); ax.set_ylabel("M COP/año")
    ax.set_title("¿Qué tan conservador planear?", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "f5_costo_vs_cuantil.png", dpi=200); plt.close(fig)

    # ---------------- Exportar para la app ----------------
    model_js = {
        "version": "1.0",
        "entrenado": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "features": FEATURES,
        "nombres": NOMBRES,
        "quantiles": QUANTILES,
        "hiperparametros": best,
        "modelos": [exportar_gbr(modelos[q]) for q in QUANTILES],
        "clim_mes": {int(k): round(float(v), 1) for k, v in clim.items()},
        "importancia": {k: round(float(v), 4) for k, v in imp.items()},
        "parametros": {"SEC": SEC, "RO_M3H": RO_M3H, "DIESEL_COP_KWH": DIESEL_COP_KWH, "AGUA_COP_M3": AGUA_COP_M3,
                       "KWP": 60, "BATERIA_KWH": 120, "L_DIESEL_KWH": 0.35},
        "metricas": {"prueba": res, "validacion": val_res, "cobertura_p10_p90": round(cobertura, 3),
                     "q_star": round(Q_STAR, 3), "ahorro_anual_cop": ahorro, "por_anio": by_year,
                     "particion": {"train": [str(tr.index[0].date()), str(tr.index[-1].date()), len(tr)],
                                   "val": [str(va.index[0].date()), str(va.index[-1].date()), len(va)],
                                   "test": [str(te.index[0].date()), str(te.index[-1].date()), len(te)]},
                     "grid_top5": grid[:5]},
    }
    # Historial de prueba para el modo "repetir un día real"
    hist = te[FEATURES].round(4).copy()
    hist["y"] = te.y_manana.round(1)
    hist["clim"] = te.clim_mes.round(1)
    model_js["historial"] = {"fechas": [str(i.date()) for i in hist.index],
                             "X": hist[FEATURES].values.tolist(), "y": hist.y.tolist(), "clim": hist.clim.tolist()}
    # Últimos 10 días del registro limpio (para la carga de CSV de ejemplo)
    ult = df.iloc[-10:][["ghi_kwh_m2", "toa_kwh_m2", "t2m_c", "rh2m_pct", "ws2m_ms", "lluvia_mm",
                         "dias_desde_limpieza", "e_fv_kwh"]]
    ult.reset_index().assign(fecha=lambda x: x.fecha.dt.strftime("%Y-%m-%d")).to_csv(
        WEB / "ejemplo_ultimos_10_dias.csv", index=False)
    # Verificación de exportación: predicciones python de referencia
    model_js["verificacion"] = {"X": te[FEATURES].iloc[:5].values.tolist(),
                                "p50_python": modelos[0.5].predict(te[FEATURES].iloc[:5]).round(3).tolist()}
    WEB.mkdir(exist_ok=True)
    (WEB / "modelo.js").write_text("window.CARBOOST_MODEL = " + json.dumps(model_js, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (ROOT / "reports" / "metricas.json").write_text(json.dumps(model_js["metricas"], ensure_ascii=False, indent=2))
    print("q* =", round(Q_STAR, 3), "| ahorro anual vs persistencia COP", f"{ahorro:,.0f}")
    print("Importancia:", imp.round(3).head(8).to_dict())


if __name__ == "__main__":
    main()
