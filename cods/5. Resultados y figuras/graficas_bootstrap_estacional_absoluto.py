import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

ruta_boot = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\bootstrap estacional\intervalos_bootstrap_estacional.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\graficas\bootstrap_estacional_relativo"
)
salida.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ruta_boot, encoding="utf-8-sig")
df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

variables = ["P_bin", "E_bin", "T_bin"]
etiquetas = {"P_bin": "P", "E_bin": "E", "T_bin": "T"}
orden = ["DJF", "MAM", "JJA", "SON"]

# ==============================
# NORMALIZAR POR AUTOFUJO DEL DESTINO
# ==============================

auto = df[df["source"] == df["target"]].copy()

auto = auto[["season", "target", "media_boot"]].rename(
    columns={
        "target": "destino",
        "media_boot": "autoflujo_destino"
    }
)

df = df.merge(
    auto,
    left_on=["season", "target"],
    right_on=["season", "destino"],
    how="left"
)

df["media_rel"] = df["media_boot"] / df["autoflujo_destino"]
df["q025_rel"] = df["q025"] / df["autoflujo_destino"]
df["q975_rel"] = df["q975"] / df["autoflujo_destino"]

df["season"] = pd.Categorical(df["season"], categories=orden, ordered=True)

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]

        d = df[
            (df["source"] == source) &
            (df["target"] == target)
        ].copy()

        d = d.sort_values("season")

        x = np.arange(len(d))
        y = d["media_rel"].to_numpy()
        yerr = np.vstack([
            y - d["q025_rel"].to_numpy(),
            d["q975_rel"].to_numpy() - y
        ])

        ax.bar(x, y, yerr=yerr, capsize=5)
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle=":", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(d["season"].astype(str))
        ax.set_title(f"{etiquetas[source]} → {etiquetas[target]}")
        ax.grid(axis="y")

        if i == 2:
            ax.set_xlabel("Estación")
        if j == 0:
            ax.set_ylabel(r"$T_{i\to j}/T_{j\to j}$")

fig.suptitle(
    f"Flujos relativos con intervalo bootstrap 95% por estación climática\nEstación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_bootstrap_season_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida / f"matriz_3x3_flujos_relativos_bootstrap_season_estacion_{ESTACION_OBJETIVO}_{K_LABEL}.pdf",
    bbox_inches="tight"
)

plt.show()