import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ESTACION_OBJETIVO = 11141

ruta_mensual = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\salidas flujo mensual\K6\resumen_promedio_matriz_flujos_por_estacion_y_mes_K6.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\graficas\graficas_mensuales"
)
salida.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ruta_mensual, encoding="utf-8-sig")

df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

if df.empty:
    raise ValueError(f"No hay datos para la estación {ESTACION_OBJETIVO} en el archivo mensual.")

variables = ["P_bin", "E_bin", "T_bin"]
etiquetas = {
    "P_bin": "P",
    "E_bin": "E",
    "T_bin": "T"
}

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]

        d = df[
            (df["source"] == source) &
            (df["target"] == target)
        ].copy()

        d = (
            d.groupby("mes_num", as_index=False)["flujo_promedio_bits"]
            .mean()
            .sort_values("mes_num")
        )

        ax.plot(d["mes_num"], d["flujo_promedio_bits"], marker="o")
        ax.set_title(f"{etiquetas[source]} → {etiquetas[target]}")
        ax.set_xticks(range(1, 13))
        ax.grid(True)

        if i == 2:
            ax.set_xlabel("Mes")
        if j == 0:
            ax.set_ylabel("Flujo (bits)")

fig.suptitle(
    f"Matriz 3x3 de flujos de información por mes\nEstación {ESTACION_OBJETIVO}",
    fontsize=16
)
fig.tight_layout()

fig.savefig(
    salida / f"matriz_3x3_flujos_mensual_estacion_{ESTACION_OBJETIVO}_K6.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida / f"matriz_3x3_flujos_mensual_estacion_{ESTACION_OBJETIVO}_K6.pdf",
    bbox_inches="tight"
)

plt.show()