import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

# Umbral mínimo para considerar estable el autoflujo destino.
EPS = 1e-4

# Por debajo de esta proporción, la anotación se marca con "*".
UMBRAL_PROP_ADVERTENCIA = 0.80

# Archivo con una fila por:
# estacion_id, replica, mes_num, source, target, flujo_info
ruta_boot = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\bootstrap mensual\bootstrap_replicas_mensual.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\graficas\bootstrap_mensual_relativo"
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

# Conversión explícita a tipos numéricos.
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

df = df.dropna(
    subset=[
        "estacion_id",
        "mes_num",
        "source",
        "target",
        "replica",
        "flujo_info"
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

# Verificar meses.
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

# Mantener sólo las variables consideradas.
df = df[
    df["source"].isin(variables)
    & df["target"].isin(variables)
].copy()

if df.empty:
    raise ValueError(
        "No hay flujos correspondientes a P_bin, E_bin y T_bin."
    )

# ============================================================
# OBTENER EL AUTOFLUJO DE LA VARIABLE DESTINO
# ============================================================

auto = df[
    df["source"] == df["target"]
].copy()

auto = auto[
    [
        "estacion_id",
        "replica",
        "mes_num",
        "target",
        "flujo_info"
    ]
].rename(
    columns={
        "target": "destino",
        "flujo_info": "autoflujo_destino"
    }
)

# Comprobar que exista un único autoflujo por grupo.
columnas_clave_auto = [
    "estacion_id",
    "replica",
    "mes_num",
    "destino"
]

duplicados_auto = auto.duplicated(
    subset=columnas_clave_auto,
    keep=False
)

if duplicados_auto.any():
    ejemplos = auto.loc[
        duplicados_auto,
        columnas_clave_auto
    ].head(20)

    raise ValueError(
        "Existen autoflujos duplicados para una misma "
        "estación, réplica, mes y destino.\n\n"
        f"{ejemplos.to_string(index=False)}"
    )

# Incorporar el autoflujo destino a cada flujo cruzado.
df_rel = df.merge(
    auto,
    left_on=[
        "estacion_id",
        "replica",
        "mes_num",
        "target"
    ],
    right_on=[
        "estacion_id",
        "replica",
        "mes_num",
        "destino"
    ],
    how="left",
    validate="many_to_one"
)

# ============================================================
# CALCULAR EL FLUJO RELATIVO R_{i->j}
# ============================================================

df_rel["denominador_valido"] = (
    df_rel["autoflujo_destino"].notna()
    & np.isfinite(df_rel["autoflujo_destino"])
    & (
        df_rel["autoflujo_destino"].abs()
        > EPS
    )
)

df_rel["flujo_relativo"] = np.where(
    df_rel["denominador_valido"],
    (
        df_rel["flujo_info"]
        / df_rel["autoflujo_destino"]
    ),
    np.nan
)

df_rel["flujo_relativo"] = (
    df_rel["flujo_relativo"]
    .replace([np.inf, -np.inf], np.nan)
)

# Una réplica sólo se cuenta como válida si el cociente final
# es finito.
df_rel["replica_valida"] = np.isfinite(
    df_rel["flujo_relativo"]
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
# RESUMEN BOOTSTRAP
# ============================================================

resumen = (
    df_rel
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
        mediana_rel=(
            "flujo_relativo",
            "median"
        ),
        q025_rel=(
            "flujo_relativo",
            q025
        ),
        q25_rel=(
            "flujo_relativo",
            q25
        ),
        q75_rel=(
            "flujo_relativo",
            q75
        ),
        q975_rel=(
            "flujo_relativo",
            q975
        ),
        media_rel=(
            "flujo_relativo",
            "mean"
        ),
        sd_rel=(
            "flujo_relativo",
            "std"
        ),
        n_boot_totales=(
            "replica",
            "count"
        ),
        n_boot_validas=(
            "flujo_relativo",
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

resumen["umbral_autoflujo"] = EPS

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
    "resumen_flujos_relativos_bootstrap_mensual_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
)

resumen.to_csv(
    ruta_resumen,
    index=False,
    encoding="utf-8-sig"
)

ruta_replicas = salida / (
    "replicas_flujos_relativos_bootstrap_mensual_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
)

columnas_replicas = [
    "estacion_id",
    "replica",
    "mes_num",
    "source",
    "target",
    "flujo_info",
    "autoflujo_destino",
    "denominador_valido",
    "replica_valida",
    "flujo_relativo"
]

df_rel[
    columnas_replicas
].to_csv(
    ruta_replicas,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FUNCIÓN PARA EXTRAER CADA PAR
# ============================================================

def preparar_datos(source, target):
    d = resumen[
        (resumen["source"] == source)
        & (resumen["target"] == target)
    ].copy()

    return d.sort_values("mes_num")


# ============================================================
# FIGURA COMPLETA:
# MEDIANA + IQR + IC 95 % + % DE RÉPLICAS VÁLIDAS
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

        mediana = d["mediana_rel"].to_numpy(dtype=float)

        p025 = d["q025_rel"].to_numpy(dtype=float)
        p25 = d["q25_rel"].to_numpy(dtype=float)
        p75 = d["q75_rel"].to_numpy(dtype=float)
        p975 = d["q975_rel"].to_numpy(dtype=float)

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

        # ----------------------------------------------------
        # Banda clara: intervalo percentil del 95 %
        # ----------------------------------------------------

        ax.fill_between(
            x[validos_ic95],
            p025[validos_ic95],
            p975[validos_ic95],
            alpha=0.18,
            label="Intervalo 95 %"
        )

        # ----------------------------------------------------
        # Banda más intensa: intervalo intercuartílico
        # ----------------------------------------------------

        ax.fill_between(
            x[validos_iqr],
            p25[validos_iqr],
            p75[validos_iqr],
            alpha=0.38,
            label="Intervalo intercuartílico"
        )

        # ----------------------------------------------------
        # Línea central: mediana bootstrap
        # ----------------------------------------------------

        ax.plot(
            x[validos_mediana],
            mediana[validos_mediana],
            marker="o",
            linewidth=1.8,
            label="Mediana bootstrap"
        )

        # Referencia de flujo nulo.
        ax.axhline(
            0,
            linestyle=":",
            linewidth=1
        )

        # Referencia de igualdad con el autoflujo.
        ax.axhline(
            1,
            linestyle="--",
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
                r"$T_{i\to j}/T_{j\to j}$"
            )

        # ----------------------------------------------------
        # ANOTAR PORCENTAJE DE RÉPLICAS VÁLIDAS
        # ----------------------------------------------------
        #
        # Se utiliza una transformación mixta:
        # x está en coordenadas de datos;
        # y está en coordenadas del eje.
        #
        # Así, las etiquetas permanecen en la parte inferior
        # del panel independientemente del rango vertical.
        # ----------------------------------------------------

        for mes, proporcion in zip(x, porcentajes):

            if not np.isfinite(proporcion):
                texto = "NA"
            elif proporcion < 0.05:
                texto = "0 %"
            else:
                texto = f"{proporcion:.0f} %"

            # Marcar los casos con menos del 80 %.
            if (
                np.isfinite(proporcion)
                and proporcion
                < 100 * UMBRAL_PROP_ADVERTENCIA
            ):
                texto = texto + "*"

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

        # Texto explicativo dentro de la primera gráfica.
        if i == 0 and j == 0:
            ax.text(
                0.02,
                0.05,
                "% válidas",
                transform=ax.transAxes,
                fontsize=7,
                va="bottom"
            )

# Leyenda compartida.
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
    "Flujos relativos mensuales: mediana, dispersión "
    "y réplicas válidas\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.text(
    0.5,
    0.035,
    (
        "Banda oscura: percentiles 25–75; "
        "banda clara: percentiles 2.5–97.5; "
        f"* menos del {100 * UMBRAL_PROP_ADVERTENCIA:.0f} % "
        "de réplicas válidas."
    ),
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.07, 1, 0.94]
)

ruta_figura_png = salida / (
    "matriz_3x3_flujos_relativos_mediana_IQR_IC95_"
    f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png"
)

ruta_figura_pdf = salida / (
    "matriz_3x3_flujos_relativos_mediana_IQR_IC95_"
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
# DIAGNÓSTICO EN CONSOLA
# ============================================================

diagnostico = resumen[
    [
        "mes_num",
        "source",
        "target",
        "n_boot_totales",
        "n_boot_validas",
        "porcentaje_boot_validas",
        "mediana_rel",
        "q25_rel",
        "q75_rel",
        "q025_rel",
        "q975_rel"
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
print("FLUJOS RELATIVOS BOOTSTRAP")
print("=" * 70)
print(f"Estación: {ESTACION_OBJETIVO}")
print(f"Etiqueta: {K_LABEL}")
print(f"Umbral del autoflujo: {EPS:.1e}")
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
            "mediana_rel": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q25_rel": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q75_rel": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q025_rel": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            ),
            "q975_rel": (
                lambda z: f"{z:.6g}"
                if pd.notna(z)
                else "NaN"
            )
        }
    )
)