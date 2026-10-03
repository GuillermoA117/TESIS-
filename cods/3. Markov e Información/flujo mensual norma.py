import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

ruta_mensual = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\salidas flujo mensual\K6\resumen_promedio_matriz_flujos_por_estacion_y_mes_K6.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\graficas\graficas_mensuales_norm"
)
salida.mkdir(parents=True, exist_ok=True)

# ==========================================
# CARGA Y FILTRO
# ==========================================

df = pd.read_csv(ruta_mensual, encoding="utf-8-sig")

df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

if df.empty:
    raise ValueError(f"No hay datos para la estación {ESTACION_OBJETIVO}.")

# ==========================================
# NORMALIZACIÓN T_{i->j} / T_{j->j}
# ==========================================

variables = ["P_bin", "E_bin", "T_bin"]

# Autoflujos del destino: T_{j->j}
auto = df[df["source"] == df["target"]].copy()

auto = auto[["mes_num", "target", "flujo_promedio_bits"]].rename(
    columns={
        "target": "destino",
        "flujo_promedio_bits": "autoflujo_destino"
    }
)

# Unir cada flujo i->j con el autoflujo j->j del mismo mes
df_norm = df.merge(
    auto,
    left_on=["mes_num", "target"],
    right_on=["mes_num", "destino"],
    how="left"
)

df_norm["flujo_relativo"] = np.where(
    df_norm["autoflujo_destino"] > 0,
    df_norm["flujo_promedio_bits"] / df_norm["autoflujo_destino"],
    np.nan
)

# Guardar tabla normalizada
df_norm.to_csv(
    salida / f"flujos_normalizados_mensual_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# FIGURA 3x3 MENSUAL NORMALIZADA
# ==========================================

etiquetas = {
    "P_bin": "P",
    "E_bin": "E",
    "T_bin": "T"
}

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]

        d = df_norm[
            (df_norm["source"] == source) &
            (df_norm["target"] == target)
        ].copy()

        d = d.sort_values("mes_num")

        ax.plot(d["mes_num"], d["flujo_relativo"], marker="o")
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.set_title(f"{etiquetas[source]} → {etiquetas[target]}")
        ax.set_xticks(range(1, 13))
        ax.grid(True)

        if i == 2:
            ax.set_xlabel("Mes")
        if j == 0:
            ax.set_ylabel(r"$T_{i\to j}/T_{j\to j}$")

fig.suptitle(
    f"Flujos relativos por mes: $T_{{i\\to j}}/T_{{j\\to j}}$\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_mensual_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_mensual_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.pdf",
    bbox_inches="tight"
)

plt.show()