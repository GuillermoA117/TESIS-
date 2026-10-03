import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

ruta_season = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\salidas flujo estacional\K6\resumen_promedio_matriz_flujos_por_estacion_y_season_K6.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\graficas\graficas_estacionales_norm"
)
salida.mkdir(parents=True, exist_ok=True)

# ==========================================
# CARGA Y FILTRO
# ==========================================

df = pd.read_csv(ruta_season, encoding="utf-8-sig")

df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

if df.empty:
    raise ValueError(f"No hay datos para la estación {ESTACION_OBJETIVO}.")

# ==========================================
# NORMALIZACIÓN T_{i->j} / T_{j->j}
# ==========================================

variables = ["P_bin", "E_bin", "T_bin"]

auto = df[df["source"] == df["target"]].copy()

auto = auto[["season", "target", "flujo_promedio_bits"]].rename(
    columns={
        "target": "destino",
        "flujo_promedio_bits": "autoflujo_destino"
    }
)

df_norm = df.merge(
    auto,
    left_on=["season", "target"],
    right_on=["season", "destino"],
    how="left"
)

df_norm["flujo_relativo"] = np.where(
    df_norm["autoflujo_destino"] > 0,
    df_norm["flujo_promedio_bits"] / df_norm["autoflujo_destino"],
    np.nan
)

df_norm.to_csv(
    salida / f"flujos_normalizados_season_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# FIGURA 3x3 ESTACIONAL NORMALIZADA
# ==========================================

etiquetas = {
    "P_bin": "P",
    "E_bin": "E",
    "T_bin": "T"
}

orden = ["DJF", "MAM", "JJA", "SON"]

df_norm["season"] = pd.Categorical(
    df_norm["season"],
    categories=orden,
    ordered=True
)

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]

        d = df_norm[
            (df_norm["source"] == source) &
            (df_norm["target"] == target)
        ].copy()

        d = d.sort_values("season")

        ax.bar(d["season"].astype(str), d["flujo_relativo"])
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.set_title(f"{etiquetas[source]} → {etiquetas[target]}")
        ax.grid(axis="y")

        if i == 2:
            ax.set_xlabel("Estación")
        if j == 0:
            ax.set_ylabel(r"$T_{i\to j}/T_{j\to j}$")

fig.suptitle(
    f"Flujos relativos por estación climática: $T_{{i\\to j}}/T_{{j\\to j}}$\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_season_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_season_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.pdf",
    bbox_inches="tight"
)

plt.show()