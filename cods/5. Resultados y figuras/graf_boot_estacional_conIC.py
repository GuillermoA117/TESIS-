import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================

ESTACION_OBJETIVO = 11141
K_LABEL = "K6"

# Umbral mínimo permitido para el autoflujo del destino.
# Las réplicas con |T_{j->j}| <= EPS se consideran inestables
# y no se utilizan para calcular el flujo relativo.
EPS = 1e-4

# Umbral mínimo recomendado de cobertura bootstrap.
# Se marcarán en las gráficas los casos con menos del 80 %
# de réplicas válidas.
PROP_MIN_RECOMENDADA = 0.80

# Archivo con una fila por réplica, estación climática,
# variable fuente y variable destino.
ruta_boot = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\bootstrap estacional\bootstrap_replicas_estacional.csv"
)

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet"
    r"\graficas\bootstrap_estacional_relativo"
)
salida.mkdir(parents=True, exist_ok=True)

# ==========================================
# CARGA
# ==========================================

df = pd.read_csv(
    ruta_boot,
    encoding="utf-8-sig"
)

columnas_requeridas = {
    "estacion_id",
    "season",
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

# Conversión numérica defensiva
df["estacion_id"] = pd.to_numeric(
    df["estacion_id"],
    errors="coerce"
)

df["replica"] = pd.to_numeric(
    df["replica"],
    errors="coerce"
)

df["flujo_info"] = pd.to_numeric(
    df["flujo_info"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "estacion_id",
        "season",
        "source",
        "target",
        "replica",
        "flujo_info"
    ]
).copy()

df["estacion_id"] = df["estacion_id"].astype(int)
df["replica"] = df["replica"].astype(int)

df = df[
    df["estacion_id"] == ESTACION_OBJETIVO
].copy()

if df.empty:
    raise ValueError(
        f"No hay datos para la estación {ESTACION_OBJETIVO}."
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

orden = [
    "DJF",
    "MAM",
    "JJA",
    "SON"
]

# Número total de réplicas bootstrap presentes
n_total_boot = df["replica"].nunique()

if n_total_boot == 0:
    raise ValueError(
        "No se encontraron réplicas bootstrap válidas."
    )

print("Estación:", ESTACION_OBJETIVO)
print("Réplicas bootstrap detectadas:", n_total_boot)
print("Umbral EPS:", EPS)

# ==========================================
# VALIDACIÓN DE DUPLICADOS
# ==========================================

columnas_clave = [
    "estacion_id",
    "replica",
    "season",
    "source",
    "target"
]

duplicados = df.duplicated(
    subset=columnas_clave,
    keep=False
)

if duplicados.any():
    ejemplo = df.loc[
        duplicados,
        columnas_clave
    ].head()

    raise ValueError(
        "Existen filas duplicadas para una misma combinación "
        "de estación, réplica, season, source y target.\n"
        f"Ejemplo:\n{ejemplo}"
    )

# ==========================================
# AUTOFUJOS DEL DESTINO
# ==========================================

# Para normalizar T_{i->j}, se necesita T_{j->j}
# de la misma réplica y estación climática.
auto = df[
    df["source"] == df["target"]
].copy()

auto = auto[
    [
        "estacion_id",
        "replica",
        "season",
        "target",
        "flujo_info"
    ]
].rename(
    columns={
        "target": "destino",
        "flujo_info": "autoflujo_destino"
    }
)

if auto.empty:
    raise ValueError(
        "No se encontraron autoflujos source == target."
    )

# ==========================================
# COCIENTE EN CADA RÉPLICA
# ==========================================

df_rel = df.merge(
    auto,
    left_on=[
        "estacion_id",
        "replica",
        "season",
        "target"
    ],
    right_on=[
        "estacion_id",
        "replica",
        "season",
        "destino"
    ],
    how="left",
    validate="many_to_one"
)

# Una réplica se considera válida cuando:
# 1. existe el autoflujo del destino;
# 2. es finito;
# 3. su valor absoluto supera EPS.
df_rel["replica_valida"] = (
    df_rel["autoflujo_destino"].notna()
    & np.isfinite(df_rel["autoflujo_destino"])
    & (
        df_rel["autoflujo_destino"].abs()
        > EPS
    )
)

df_rel["flujo_relativo"] = np.where(
    df_rel["replica_valida"],
    (
        df_rel["flujo_info"]
        / df_rel["autoflujo_destino"]
    ),
    np.nan
)

df_rel = df_rel.replace(
    [np.inf, -np.inf],
    np.nan
)

# Guardar también las réplicas relativas para inspección posterior
df_rel.to_csv(
    salida
    / (
        "bootstrap_replicas_relativas_estacional_"
        f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# RESUMEN BOOTSTRAP DEL COCIENTE
# ==========================================

resumen = (
    df_rel
    .groupby(
        [
            "estacion_id",
            "season",
            "source",
            "target"
        ],
        as_index=False
    )
    .agg(
        media_rel=(
            "flujo_relativo",
            "mean"
        ),
        mediana_rel=(
            "flujo_relativo",
            "median"
        ),
        sd_rel=(
            "flujo_relativo",
            "std"
        ),
        q025_rel=(
            "flujo_relativo",
            lambda x: x.quantile(0.025)
        ),
        q500_rel=(
            "flujo_relativo",
            lambda x: x.quantile(0.500)
        ),
        q975_rel=(
            "flujo_relativo",
            lambda x: x.quantile(0.975)
        ),
        n_boot_validas=(
            "flujo_relativo",
            "count"
        ),
        autoflujo_medio=(
            "autoflujo_destino",
            "mean"
        ),
        autoflujo_mediano=(
            "autoflujo_destino",
            "median"
        ),
        autoflujo_minimo=(
            "autoflujo_destino",
            "min"
        ),
        autoflujo_maximo=(
            "autoflujo_destino",
            "max"
        )
    )
)

# Número total de réplicas esperadas por combinación
resumen["n_boot_totales"] = n_total_boot

# Proporción de réplicas válidas
resumen["prop_boot_validas"] = (
    resumen["n_boot_validas"]
    / resumen["n_boot_totales"]
)

# Indicador de cobertura
resumen["cobertura_bootstrap"] = np.where(
    resumen["prop_boot_validas"]
    >= PROP_MIN_RECOMENDADA,
    "adecuada",
    "baja"
)

# Orden estacional
resumen["season"] = pd.Categorical(
    resumen["season"],
    categories=orden,
    ordered=True
)

resumen = resumen.sort_values(
    [
        "source",
        "target",
        "season"
    ]
).reset_index(drop=True)

# Guardar resumen completo
ruta_resumen = (
    salida
    / (
        "resumen_flujos_relativos_bootstrap_estacional_"
        f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
    )
)

resumen.to_csv(
    ruta_resumen,
    index=False,
    encoding="utf-8-sig"
)

# Guardar un diagnóstico separado de cobertura
diagnostico = resumen[
    [
        "estacion_id",
        "season",
        "source",
        "target",
        "n_boot_validas",
        "n_boot_totales",
        "prop_boot_validas",
        "cobertura_bootstrap",
        "autoflujo_medio",
        "autoflujo_mediano",
        "autoflujo_minimo",
        "autoflujo_maximo"
    ]
].copy()

diagnostico.to_csv(
    salida
    / (
        "diagnostico_cobertura_bootstrap_estacional_"
        f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# FUNCIÓN AUXILIAR
# ==========================================

def preparar_datos(
    source: str,
    target: str
) -> pd.DataFrame:
    """
    Devuelve los resultados de un flujo source -> target
    ordenados según DJF, MAM, JJA y SON.
    """

    d = resumen[
        (resumen["source"] == source)
        & (resumen["target"] == target)
    ].copy()

    # Reindexar para garantizar que aparezcan las cuatro estaciones
    d = (
        d
        .set_index("season")
        .reindex(orden)
        .reset_index()
    )

    return d


def marcar_baja_cobertura(
    ax: plt.Axes,
    d: pd.DataFrame
) -> None:
    """
    Añade una advertencia al panel cuando al menos una estación
    climática tiene menos de PROP_MIN_RECOMENDADA réplicas válidas.
    """

    cobertura = pd.to_numeric(
        d["prop_boot_validas"],
        errors="coerce"
    )

    if (
        cobertura.notna().any()
        and cobertura.min()
        < PROP_MIN_RECOMENDADA
    ):
        ax.text(
            0.03,
            0.95,
            "Baja cobertura\nbootstrap",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round",
                "alpha": 0.20
            }
        )


def anotar_cobertura(
    ax: plt.Axes,
    x: np.ndarray,
    d: pd.DataFrame
) -> None:
    """
    Anota debajo de cada barra el porcentaje de réplicas válidas.
    """

    proporciones = pd.to_numeric(
        d["prop_boot_validas"],
        errors="coerce"
    ).to_numpy(dtype=float)

    for posicion, proporcion in zip(
        x,
        proporciones
    ):
        if np.isfinite(proporcion):
            ax.text(
                posicion,
                0.02,
                f"{100 * proporcion:.0f} %",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

# ==========================================
# FIGURA 1:
# MEDIANA BOOTSTRAP DEL FLUJO RELATIVO
# ==========================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(14, 10),
    sharey=False
)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]
        d = preparar_datos(
            source,
            target
        )

        x = np.arange(len(orden))

        # Se usa la mediana bootstrap como estimador central
        y = pd.to_numeric(
            d["mediana_rel"],
            errors="coerce"
        ).to_numpy(dtype=float)

        valores_validos = np.isfinite(y)

        ax.bar(
            x[valores_validos],
            y[valores_validos]
        )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1
        )

        ax.axhline(
            1,
            linestyle="--",
            linewidth=1
        )

        ax.set_xticks(x)
        ax.set_xticklabels(orden)

        ax.set_title(
            f"{etiquetas[source]} → "
            f"{etiquetas[target]}"
        )

        ax.grid(
            axis="y",
            alpha=0.5
        )

        marcar_baja_cobertura(
            ax,
            d
        )

        anotar_cobertura(
            ax,
            x,
            d
        )

        if i == 2:
            ax.set_xlabel(
                "Estación climática"
            )

        if j == 0:
            ax.set_ylabel(
                r"$T_{i\to j}/T_{j\to j}$"
            )

fig.suptitle(
    "Mediana bootstrap del flujo relativo "
    "por estación climática\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

fig.savefig(
    salida
    / (
        "matriz_3x3_mediana_flujos_relativos_estacional_"
        f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida
    / (
        "matriz_3x3_mediana_flujos_relativos_estacional_"
        f"estacion_{ESTACION_OBJETIVO}_{K_LABEL}.pdf"
    ),
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================
# FIGURA 2:
# MEDIANA + INTERVALO BOOTSTRAP DEL 95 %
# ==========================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(14, 10),
    sharey=False
)

for i, source in enumerate(variables):
    for j, target in enumerate(variables):
        ax = axes[i, j]
        d = preparar_datos(
            source,
            target
        )

        x = np.arange(len(orden))

        # Estimador central: mediana bootstrap
        y = pd.to_numeric(
            d["mediana_rel"],
            errors="coerce"
        ).to_numpy(dtype=float)

        q025 = pd.to_numeric(
            d["q025_rel"],
            errors="coerce"
        ).to_numpy(dtype=float)

        q975 = pd.to_numeric(
            d["q975_rel"],
            errors="coerce"
        ).to_numpy(dtype=float)

        valores_validos = (
            np.isfinite(y)
            & np.isfinite(q025)
            & np.isfinite(q975)
        )

        x_validos = x[valores_validos]
        y_validos = y[valores_validos]
        q025_validos = q025[valores_validos]
        q975_validos = q975[valores_validos]

        # Errores asimétricos respecto a la mediana
        error_inferior = (
            y_validos
            - q025_validos
        )

        error_superior = (
            q975_validos
            - y_validos
        )

        # Matplotlib requiere errores no negativos.
        # np.maximum protege contra pequeñas diferencias numéricas.
        error_inferior = np.maximum(
            error_inferior,
            0
        )

        error_superior = np.maximum(
            error_superior,
            0
        )

        yerr = np.vstack(
            [
                error_inferior,
                error_superior
            ]
        )

        ax.bar(
            x_validos,
            y_validos,
            yerr=yerr,
            capsize=6
        )

        ax.axhline(
            0,
            linestyle=":",
            linewidth=1
        )

        ax.axhline(
            1,
            linestyle="--",
            linewidth=1
        )

        ax.set_xticks(x)
        ax.set_xticklabels(orden)

        ax.set_title(
            f"{etiquetas[source]} → "
            f"{etiquetas[target]}"
        )

        ax.grid(
            axis="y",
            alpha=0.5
        )

        marcar_baja_cobertura(
            ax,
            d
        )

        anotar_cobertura(
            ax,
            x,
            d
        )

        if i == 2:
            ax.set_xlabel(
                "Estación climática"
            )

        if j == 0:
            ax.set_ylabel(
                r"$T_{i\to j}/T_{j\to j}$"
            )

fig.suptitle(
    "Mediana bootstrap del flujo relativo "
    "con intervalo percentil del 95 %\n"
    f"Estación {ESTACION_OBJETIVO}",
    fontsize=16
)

fig.tight_layout(
    rect=[0, 0, 1, 0.94]
)

fig.savefig(
    salida
    / (
        "matriz_3x3_mediana_flujos_relativos_IC95_"
        f"estacional_estacion_{ESTACION_OBJETIVO}_"
        f"{K_LABEL}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    salida
    / (
        "matriz_3x3_mediana_flujos_relativos_IC95_"
        f"estacional_estacion_{ESTACION_OBJETIVO}_"
        f"{K_LABEL}.pdf"
    ),
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RESUMEN EN CONSOLA
# ==========================================

print()
print("Proceso terminado.")
print("Resultados guardados en:", salida)
print("Archivo resumen:", ruta_resumen)
print()

print(
    "Casos con cobertura bootstrap inferior a "
    f"{100 * PROP_MIN_RECOMENDADA:.0f} %:"
)

casos_baja_cobertura = resumen[
    resumen["prop_boot_validas"]
    < PROP_MIN_RECOMENDADA
][
    [
        "season",
        "source",
        "target",
        "n_boot_validas",
        "n_boot_totales",
        "prop_boot_validas",
        "autoflujo_mediano"
    ]
].copy()

if casos_baja_cobertura.empty:
    print("Ninguno.")
else:
    print(
        casos_baja_cobertura.to_string(
            index=False
        )
    )