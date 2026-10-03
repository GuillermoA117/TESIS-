import numpy as np
import pandas as pd
from pathlib import Path
from kgada2diarios import matriz_flujos_observables

# ==========================================
# CONFIGURACIÓN
# ==========================================

ruta = r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\base_diaria_discretizada_mixta_PET.csv"
ESTACION_OBJETIVO = 11141
salida = Path(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\salidas flujo mensual\K6")
salida.mkdir(parents=True, exist_ok=True)

# ==========================================
# CARGA
# ==========================================

df = pd.read_csv(ruta, encoding="utf-8-sig")
df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

df = df.dropna(subset=["fecha", "estacion_id", "P_bin", "E_bin", "T_bin"]).copy()

# Asegurar enteros
df["estacion_id"] = pd.to_numeric(df["estacion_id"], errors="coerce").astype("Int64")
df["P_bin"] = pd.to_numeric(df["P_bin"], errors="coerce").astype("Int64")
df["E_bin"] = pd.to_numeric(df["E_bin"], errors="coerce").astype("Int64")
df["T_bin"] = pd.to_numeric(df["T_bin"], errors="coerce").astype("Int64")

# ==========================================
# CARGAR ESTACIONES REPRESENTATIVAS
# ==========================================
df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

print("Estación usada:", ESTACION_OBJETIVO)
print("Número de registros:", len(df))

print("Estaciones representativas usadas:")
print(sorted(df["estacion_id"].dropna().unique()))
print("Número de estaciones usadas:", df["estacion_id"].nunique())

# ==========================================
# VARIABLES TEMPORALES: MES DEL AÑO
# ==========================================

df["anio"] = df["fecha"].dt.year
df["mes_num"] = df["fecha"].dt.month

# Ordenar correctamente
df = df.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)

# ==========================================
# MATRIZ COMPLETA DE FLUJOS MULTIVARIADOS
# ==========================================

matriz_flujos = matriz_flujos_observables(
    df=df,
    observable_cols=("P_bin", "E_bin", "T_bin"),
    group_cols=["estacion_id", "mes_num"],
    time_col="fecha",
    base=2
)

matriz_flujos.to_csv(
    salida / "matriz_flujos_multivariados_PET_por_mes_del_anio_K6.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# RESUMEN PROMEDIO POR ESTACIÓN, MES, SOURCE Y TARGET
# ==========================================

col_flujo = None
for c in matriz_flujos.columns:
    if "flujo" in c.lower() or "transfer" in c.lower() or "info" in c.lower():
        col_flujo = c
        break

if col_flujo is not None:
    resumen_mensual = (
        matriz_flujos
        .groupby(["estacion_id", "mes_num", "source", "target"], as_index=False)[col_flujo]
        .mean()
        .rename(columns={col_flujo: "flujo_promedio_bits"})
    )

    resumen_mensual.to_csv(
        salida / "resumen_promedio_matriz_flujos_por_estacion_y_mes_K6.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("Listo. Archivos guardados en:", salida)
print("Columnas del resultado:", list(matriz_flujos.columns))
print(matriz_flujos.head())