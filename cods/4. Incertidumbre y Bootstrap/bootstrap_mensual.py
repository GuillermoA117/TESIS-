import numpy as np
import pandas as pd
from pathlib import Path
from kgada2diarios import matriz_flujos_observables

# ==========================================
# CONFIGURACIÓN
# ==========================================

ruta = r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\base_diaria_discretizada_mixta_PET.csv"

ESTACION_OBJETIVO = 11141
N_BOOT = 300
L_BLOQUE = 30
SEED = 42

salida = Path(
    r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\chet\bootstrap mensual"
)
salida.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ==========================================
# FUNCIONES
# ==========================================

def bootstrap_bloques(df, L, rng):
    """
    Block bootstrap con bloques consecutivos.
    Usa boot_fecha artificial para conservar orden interno
    y romper transiciones entre bloques.
    """
    df = df.sort_values("fecha").reset_index(drop=True).copy()
    n = len(df)

    if n <= L:
        raise ValueError("La longitud del bloque es mayor o igual al número de datos.")

    n_bloques = int(np.ceil(n / L))
    partes = []
    dia_actual = pd.Timestamp("2000-01-01")

    for _ in range(n_bloques):
        inicio = rng.integers(0, n - L)
        bloque = df.iloc[inicio:inicio + L].copy()

        bloque["boot_fecha"] = pd.date_range(
            start=dia_actual,
            periods=len(bloque),
            freq="D"
        )

        # hueco artificial para evitar transición entre bloques
        dia_actual = dia_actual + pd.Timedelta(days=L + 3)

        partes.append(bloque)

    out = pd.concat(partes, ignore_index=True)
    return out.iloc[:n].copy()


# ==========================================
# CARGA
# ==========================================

df = pd.read_csv(ruta, encoding="utf-8-sig")
df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

df = df.dropna(subset=["fecha", "estacion_id", "P_bin", "E_bin", "T_bin"]).copy()

df["estacion_id"] = pd.to_numeric(df["estacion_id"], errors="coerce").astype("Int64")
df["P_bin"] = pd.to_numeric(df["P_bin"], errors="coerce").astype("Int64")
df["E_bin"] = pd.to_numeric(df["E_bin"], errors="coerce").astype("Int64")
df["T_bin"] = pd.to_numeric(df["T_bin"], errors="coerce").astype("Int64")

df = df[df["estacion_id"] == ESTACION_OBJETIVO].copy()

if df.empty:
    raise ValueError(f"No hay datos para la estación {ESTACION_OBJETIVO}")

df["anio"] = df["fecha"].dt.year
df["mes_num"] = df["fecha"].dt.month

df = df.sort_values("fecha").reset_index(drop=True)

print("Estación:", ESTACION_OBJETIVO)
print("Registros:", len(df))

# ==========================================
# FLUJO OBSERVADO
# ==========================================

matriz_obs = matriz_flujos_observables(
    df=df,
    observable_cols=("P_bin", "E_bin", "T_bin"),
    group_cols=["estacion_id", "mes_num"],
    time_col="fecha",
    base=2
)

matriz_obs.to_csv(
    salida / "flujo_observado_mensual.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# BOOTSTRAP
# ==========================================

resultados = []

for b in range(N_BOOT):
    print(f"Bootstrap mensual {b + 1}/{N_BOOT}")

    df_boot = bootstrap_bloques(df, L_BLOQUE, rng)

    matriz_boot = matriz_flujos_observables(
        df=df_boot,
        observable_cols=("P_bin", "E_bin", "T_bin"),
        group_cols=["estacion_id", "mes_num"],
        time_col="boot_fecha",
        base=2
    )

    matriz_boot["replica"] = b
    resultados.append(matriz_boot)

boot = pd.concat(resultados, ignore_index=True)

boot.to_csv(
    salida / "bootstrap_replicas_mensual.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# INTERVALOS DE CONFIANZA
# ==========================================

ic = (
    boot
    .groupby(["estacion_id", "mes_num", "source", "target"])["flujo_info"]
    .agg(
        media_boot="mean",
        sd_boot="std",
        q025=lambda x: x.quantile(0.025),
        q500=lambda x: x.quantile(0.500),
        q975=lambda x: x.quantile(0.975)
    )
    .reset_index()
)

ic.to_csv(
    salida / "intervalos_bootstrap_mensual.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Listo. Resultados guardados en:", salida)