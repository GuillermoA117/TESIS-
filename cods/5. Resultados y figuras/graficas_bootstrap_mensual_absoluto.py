import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

# Archivo con una fila por réplica, mes, source y target.
ruta_boot = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\bootstrap mensual\bootstrap_replicas_mensual.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\graficas\bootstrap_mensual_absoluto"
)

salida.mkdir(parents=True, exist_ok=True)

# ============================================================
# CARGA Y VALIDACIÓN
# ============================================================

df = pd.read_csv(
    ruta_boot,
    encoding="utf-8-sig"
)

columnas_requeridas = {
    "estacion_id",
    "mes_num",
    "source",
    "target",
    "replica",
    "flujo_info"
}

faltantes = columnas_requeridas.difference(df.columns)

if faltantes:
    raise ValueError(
        "Faltan columnas en el archivo bootstrap: "
        f"{sorted(faltantes)}"
    )

# Conversiones numéricas.
for columna in [
    "estacion_id",
    "mes_num",
    "replica",
    "flujo_info"
]:
    df[columna] = pd.to_numeric(
        df[columna],
        errors="coerce"
    )

# Para contabilizar réplicas totales se conserva inicialmente
# la información aunque flujo_info sea NaN.
df = df.dropna(
    subset=[
        "estacion_id",
        "mes_num",
        "source",
        "target",
        "replica"
    ]
).copy()

df["estacion_id"] = df["estacion_id"].astype(int)
df["mes_num"] = df["mes_num"].astype(int)
df["replica"] = df["replica"].astype(int)

# Filtrar estación.
df = df[
    df["estacion_id"] == ESTACION_OBJETIVO
].copy()

if df.empty:
    raise ValueError(
        f"No hay datos para la estación {ESTACION_OBJETIVO}."
    )

# Verificación temporal.
meses_invalidos = sorted(
    df.loc[
        ~df["mes_num"].between(1, 12),
        "mes_num"
    ].unique()
)

if meses_invalidos:
    raise ValueError(
        "Se encontraron meses fuera del intervalo 1--12: "
        f"{meses_invalidos}"
    )

variables = [
    "P_bin",
    "E_bin",
    "T_bin"
]

etiquetas = {
    "P_bin": "P",
    "E_bin": "E",
    "T_bin": "T"
}

df = df[
    df["source"].isin(variables)
    & df["target"].isin(variables)
].copy()

if df.empty:
    raise ValueError(
        "No hay flujos correspondientes a P_bin, E_bin y T_bin."
    )

# Sustituir infinitos por NaN.
df["flujo_info"] = df["flujo_info"].replace(
    [np.inf, -np.inf],
    np.nan
)

df["replica_valida"] = np.isfinite(
    df["flujo_info"]
)

# ============================================================
# FUNCIONES DE RESUMEN
# ============================================================

def q025(x):
    return x.quantile(0.025)


def q25(x):
    return x.quantile(0.25)


def q75(x):
    return x.quantile(0.75)


def q975(x):
    return x.quantile(0.975)


# ============================================================
# RESUMEN BOOTSTRAP DEL FLUJO ABSOLUTO
# ============================================================

resumen = (
    df
    .groupby(
        [
            "estacion_id",
            "mes_num",
            "source",
            "target"
        ],
        as_index=False
    )
    .agg(
        mediana_flujo=(
            "flujo_info",
            "median"
        ),
        q025_flujo=(
            "flujo_info",
            q025
        ),
        q25_flujo=(
            "flujo_info",
            q25
        ),
        q75_flujo=(
            "flujo_info",
            q75
        ),
        q975_flujo=(
            "flujo_info",
            q975
        ),
        media_flujo=(
            "flujo_info",
            "mean"
        ),
        sd_flujo=(
            "flujo_info",
            "std"
        ),
        n_boot_totales=(
            "replica",
            "count"
        ),
        n_boot_validas=(
            "flujo_info",
            "count"
        )
    )
)

resumen["prop_boot_validas"] = np.where(
    resumen["n_boot_totales"] > 0,
    (
        resumen["n_boot_validas"]
        / resumen["n_boot_totales"]
    ),
    np.nan
)

resumen["porcentaje_boot_validas"] = (
    100.0 * resumen["prop_boot_validas"]
)

resumen = resumen.sort_values(
    [
        "source",
        "target",
        "mes_num"
    ]
).reset_index(drop=True)

# ============================================================
# GUARDAR TABLAS
# ============================================================

ruta_resumen = salida / (
    "resumen_flujos_absolutos_bootstrap_mensual_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
)

resumen.to_csv(
    ruta_resumen,
    index=False,
    encoding="utf-8-sig"
)

ruta_replicas = salida / (
    "replicas_flujos_absolutos_bootstrap_mensual_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
)

df[
    [
        "estacion_id",
        "replica",
        "mes_num",
        "source",
        "target",
        "flujo_info",
        "replica_valida"
    ]
].to_csv(
    ruta_replicas,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def preparar_datos(source, target):
    d = resumen[
        (resumen["source"] == source)
        & (resumen["target"] == target)
    ].copy()

    return d.sort_values("mes_num")


# ============================================================
# FIGURA:
# MEDIANA + IQR + IC 95 % DEL FLUJO ABSOLUTO
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 11),
    sharex=True,
    sharey=False
)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):

        ax = axes[i, j]
        d = preparar_datos(source, target)

        x = d["mes_num"].to_numpy(dtype=float)

        mediana = d[
            "mediana_flujo"
        ].to_numpy(dtype=float)

        p025 = d[
            "q025_flujo"
        ].to_numpy(dtype=float)

        p25 = d[
            "q25_flujo"
        ].to_numpy(dtype=float)

        p75 = d[
            "q75_flujo"
        ].to_numpy(dtype=float)

        p975 = d[
            "q975_flujo"
        ].to_numpy(dtype=float)

        porcentajes = d[
            "porcentaje_boot_validas"
        ].to_numpy(dtype=float)

        validos_ic95 = (
            np.isfinite(x)
            & np.isfinite(p025)
            & np.isfinite(p975)
        )

        validos_iqr = (
            np.isfinite(x)
            & np.isfinite(p25)
            & np.isfinite(p75)
        )

        validos_mediana = (
            np.isfinite(x)
            & np.isfinite(mediana)
        )

        # Banda exterior: intervalo del 95 %.
        ax.fill_between(
            x[validos_ic95],
            p025[validos_ic95],
            p975[validos_ic95],
            alpha=0.18,
            label="Intervalo 95 %"
        )

        # Banda interior: IQR.
        ax.fill_between(
            x[validos_iqr],
            p25[validos_iqr],
            p75[validos_iqr],
            alpha=0.38,
            label="Intervalo intercuartílico"
        )

        # Mediana.
        ax.plot(
            x[validos_mediana],
            mediana[validos_mediana],
            marker="o",
            linewidth=1.8,
            label="Mediana bootstrap"
        )

        # Flujo nulo.
        ax.axhline(
            0,
            linestyle=":",
            linewidth=1
        )

        ax.set_title(
            f"{etiquetas[source]} "
            f"→ {etiquetas[target]}"
        )

        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.45)

        if i == 2:
            ax.set_xlabel("Mes")

        if j == 0:
            ax.set_ylabel(
                r"$T_{i\to j}$ [bits]"
            )

        # Porcentaje de réplicas finitas.
        for mes, porcentaje in zip(x, porcentajes):

            if not np.isfinite(porcentaje):
                texto = "NA"
            elif porcentaje < 0.05:
                texto = "0 %"
            else:
                texto = f"{porcentaje:.0f} %"

            ax.text(
                mes,
                0.015,
                texto,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                clip_on=True
            )

        if i == 0 and j == 0:
            ax.text(
                0.02,
                0.05,
                "% válidas",
                transform=ax.transAxes,
                fontsize=7,
                va="bottom"
            )

# Leyenda común.
handles, labels = axes[0, 1].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.005),
    frameon=True
)

fig.suptitle(
    "Flujos absolutos mensuales: mediana e intervalos "
    "bootstrap\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.text(
    0.5,
    0.035,
    (
        "Banda oscura: percentiles 25–75; "
        "banda clara: percentiles 2.5–97.5. "
        "Los flujos se expresan en bits."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.07, 1, 0.94]
)

ruta_figura_png = salida / (
    "matriz_3x3_flujos_absolutos_mediana_IQR_IC95_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png"
)

ruta_figura_pdf = salida / (
    "matriz_3x3_flujos_absolutos_mediana_IQR_IC95_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.pdf"
)

fig.savefig(
    ruta_figura_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    ruta_figura_pdf,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# DIAGNÓSTICO
# ============================================================

diagnostico = resumen[
    [
        "mes_num",
        "source",
        "target",
        "n_boot_totales",
        "n_boot_validas",
        "porcentaje_boot_validas",
        "mediana_flujo",
        "q25_flujo",
        "q75_flujo",
        "q025_flujo",
        "q975_flujo"
    ]
].sort_values(
    [
        "porcentaje_boot_validas",
        "source",
        "target",
        "mes_num"
    ]
)

print("\n" + "=" * 70)
print("FLUJOS ABSOLUTOS BOOTSTRAP")
print("=" * 70)
print(f"Estación: {ESTACION_OBJETIVO}")
print(f"Etiqueta: {K_LABEL}")
print(f"Resumen guardado en: {ruta_resumen}")
print(f"Réplicas guardadas en: {ruta_replicas}")
print(f"Figura guardada en: {ruta_figura_png}")

print("\nCasos con menor proporción de réplicas válidas:\n")

print(
    diagnostico.head(30).to_string(
        index=False,
        formatters={
            "porcentaje_boot_validas": (
                lambda z: f"{z:.1f} %"
                if pd.notna(z)
                else "NaN"
            ),
            "mediana_flujo": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q25_flujo": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q75_flujo": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q025_flujo": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q975_flujo": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            )
        }
    )
)