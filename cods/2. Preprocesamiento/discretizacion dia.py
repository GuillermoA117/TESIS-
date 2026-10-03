import numpy as np
import pandas as pd
from pathlib import Path


def limpiar_numerica(s):
    return pd.to_numeric(s, errors="coerce")


def discretizar_cuantiles(serie, n_bins=2, labels=None):
    s = limpiar_numerica(serie)

    if labels is None:
        labels = list(range(1, n_bins + 1))

    no_na = s.dropna()
    if no_na.empty:
        return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    try:
        bins = pd.qcut(no_na, q=n_bins, labels=labels, duplicates="drop")
    except ValueError:
        bins = pd.qcut(
            no_na.rank(method="average"),
            q=min(n_bins, no_na.nunique()),
            labels=labels[:min(n_bins, no_na.nunique())],
            duplicates="drop"
        )

    out = pd.Series(index=s.index, dtype="object")
    out.loc[no_na.index] = bins.astype(object)
    return out.astype("Int64")


def discretizar_precipitacion_mixta(serie_precip, n_bins_lluvia=2):
    s = limpiar_numerica(serie_precip)
    out = pd.Series(index=s.index, dtype="object")

    out.loc[s == 0] = 0

    mask_pos = s > 0
    s_pos = s.loc[mask_pos]

    if s_pos.empty:
        return out.astype("Int64")

    labels = list(range(1, n_bins_lluvia + 1))

    try:
        bins_pos = pd.qcut(
            s_pos,
            q=n_bins_lluvia,
            labels=labels,
            duplicates="drop"
        )
    except ValueError:
        bins_pos = pd.qcut(
            s_pos.rank(method="average"),
            q=min(n_bins_lluvia, s_pos.nunique()),
            labels=labels[:min(n_bins_lluvia, s_pos.nunique())],
            duplicates="drop"
        )

    out.loc[mask_pos] = bins_pos.astype(object)
    return out.astype("Int64")


def construir_estado_markov(
    df,
    col_precip="lluvia_total_mm",
    col_evap="evaporacion_mm",
    col_temp="temp_media_c",
    n_bins_precip_lluvia=2,
    n_bins_evap=3,
    n_bins_temp=3
):
    d = df.copy()

    d["P_bin"] = discretizar_precipitacion_mixta(
        d[col_precip],
        n_bins_lluvia=n_bins_precip_lluvia
    )

    d["E_bin"] = discretizar_cuantiles(
        d[col_evap],
        n_bins=n_bins_evap
    )

    d["T_bin"] = discretizar_cuantiles(
        d[col_temp],
        n_bins=n_bins_temp
    )

    mask_ok = d[["P_bin", "E_bin", "T_bin"]].notna().all(axis=1)

    d["estado_markov"] = pd.NA
    d.loc[mask_ok, "estado_markov"] = (
        "P" + d.loc[mask_ok, "P_bin"].astype(str) +
        "_E" + d.loc[mask_ok, "E_bin"].astype(str) +
        "_T" + d.loc[mask_ok, "T_bin"].astype(str)
    )

    return d


def resumen_discretizacion(df, nombre_estacion=None):
    if nombre_estacion is not None:
        print(f"\n===== ESTACIÓN {nombre_estacion} =====")

    print("\n--- Distribución P_bin ---")
    print(df["P_bin"].value_counts(dropna=False).sort_index())
    print(df["P_bin"].value_counts(normalize=True, dropna=False).sort_index())

    print("\n--- Distribución E_bin ---")
    print(df["E_bin"].value_counts(dropna=False).sort_index())
    print(df["E_bin"].value_counts(normalize=True, dropna=False).sort_index())

    print("\n--- Distribución T_bin ---")
    print(df["T_bin"].value_counts(dropna=False).sort_index())
    print(df["T_bin"].value_counts(normalize=True, dropna=False).sort_index())

    print("\n--- Estados Markov más frecuentes ---")
    print(df["estado_markov"].value_counts(dropna=False).head(20))

    print(f"\nNúmero de estados observados: {df['estado_markov'].nunique(dropna=True)}")


# ============================================================
# EJEMPLO
# ============================================================

ruta = r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\salida_conagua_gto\diarios_txt\salidas_markov_diarios\03_base_estaciones_15_anios.csv"

df = pd.read_csv(ruta, encoding="utf-8-sig")
df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

df_disc = construir_estado_markov(
    df,
    col_precip="lluvia_total_mm",
    col_evap="evaporacion_mm",
    col_temp="temp_media_c",
    n_bins_precip_lluvia=2,  # estados 0,1,2
    n_bins_evap=2,
    n_bins_temp=2
)

resumen_discretizacion(df_disc)

SALIDA = Path(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet")
SALIDA.mkdir(parents=True, exist_ok=True)

df_disc.to_csv(
    SALIDA / "base_diaria_discretizada_mixta_PET.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df_disc["P_bin"].value_counts(normalize=True, dropna=False).sort_index())
print(df_disc["E_bin"].value_counts(normalize=True, dropna=False).sort_index())
print(df_disc["T_bin"].value_counts(normalize=True, dropna=False).sort_index())