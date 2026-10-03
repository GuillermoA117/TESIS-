import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_TXT = Path(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\salida_conagua_gto\diarios_txt")
SALIDA = CARPETA_TXT / "salidas_markov_diarios"
SALIDA.mkdir(exist_ok=True)

# Periodo común global para comparar estaciones
ANIO_GLOBAL_INI = 1980
ANIO_GLOBAL_FIN = 2020

# Mínimo de años completos por estación
MIN_ANIOS_COMPLETOS = 15

# Cobertura mínima anual para considerar un año "completo"
# 0.85 = al menos 85% de los días esperados del año
COBERTURA_MIN_ANUAL = 0.95

# Número de bins para discretización del estado
N_BINS = 3


# ============================================================
# UTILIDADES DE TEXTO
# ============================================================

def reparar_mojibake(texto):
    reemplazos = {
        "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó", "Ãš": "Ú", "Ã‘": "Ñ",
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú", "Ã±": "ñ",
        "Â°": "°", "Âº": "º", "Âª": "ª"
    }
    for malo, bueno in reemplazos.items():
        texto = texto.replace(malo, bueno)
    return texto


def leer_texto_robusto(ruta_txt):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            texto = ruta_txt.read_text(encoding=enc, errors="ignore")
            texto = reparar_mojibake(texto)
            return texto
        except Exception:
            pass

    texto = ruta_txt.read_text(errors="ignore")
    texto = reparar_mojibake(texto)
    return texto


def normalizar_texto(s):
    if s is None:
        return ""

    s = reparar_mojibake(str(s))
    s = s.upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# UTILIDADES NUMÉRICAS Y CLIMÁTICAS
# ============================================================

def limpiar_numero(x):
    if x is None:
        return np.nan
    x = str(x).strip()
    if x == "":
        return np.nan
    x = x.replace(",", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def estacion_climatologica(mes_num):
    if mes_num in [12, 1, 2]:
        return "DJF"
    elif mes_num in [3, 4, 5]:
        return "MAM"
    elif mes_num in [6, 7, 8]:
        return "JJA"
    elif mes_num in [9, 10, 11]:
        return "SON"
    return np.nan


def dias_esperados_en_anio(anio):
    return 366 if pd.Timestamp(year=int(anio), month=12, day=31).is_leap_year else 365


# ============================================================
# EXTRACCIÓN DE METADATA
# ============================================================

def obtener_metadata(texto):
    patrones = {
        "estacion_id": [
            r"ESTACI[ÓO]N\s*:\s*(.+)",
            r"ESTACION\s*:\s*(.+)"
        ],
        "nombre": [r"NOMBRE\s*:\s*(.+)"],
        "estado": [r"ESTADO\s*:\s*(.+)"],
        "municipio": [r"MUNICIPIO\s*:\s*(.+)"],
        "situacion": [
            r"SITUACI[ÓO]N\s*:\s*(.+)",
            r"SITUACION\s*:\s*(.+)"
        ],
        "latitud": [r"LATITUD\s*:\s*([-\d\.]+)"],
        "longitud": [r"LONGITUD\s*:\s*([-\d\.]+)"],
        "altitud_msnm": [r"ALTITUD\s*:\s*([-\d\.]+)"]
    }

    meta = {}
    for clave, lista_patrones in patrones.items():
        valor = None
        for patron in lista_patrones:
            m = re.search(patron, texto, flags=re.IGNORECASE)
            if m:
                valor = m.group(1).strip()
                break
        meta[clave] = valor

    for clave in ["latitud", "longitud", "altitud_msnm"]:
        meta[clave] = limpiar_numero(meta[clave])

    return meta


# ============================================================
# EXTRACCIÓN DE TABLA DIARIA
# ============================================================

def encontrar_header_diario(lineas):
    """
    Busca la línea del encabezado diario. Normalmente contiene FECHA.
    """
    for i, ln in enumerate(lineas):
        ln_norm = normalizar_texto(ln)
        if "FECHA" in ln_norm and (
            "PRECIP" in ln_norm or
            "EVAP" in ln_norm or
            "TMAX" in ln_norm or
            "TMIN" in ln_norm or
            "TEMPERATURA" in ln_norm
        ):
            return i
    return None


def parsear_tabla_diaria(texto):
    """
    Intenta parsear la tabla diaria desde la línea que contiene FECHA.
    Espera algo del tipo:
    FECHA PRECIP EVAP TMAX TMIN ...
    """
    lineas = [ln.rstrip() for ln in texto.splitlines() if ln.strip()]
    idx_header = encontrar_header_diario(lineas)

    if idx_header is None:
        return None

    header = re.split(r"\s+", lineas[idx_header].strip())
    header = [normalizar_texto(h) for h in header]

    # limpiar nombres de columnas
    header_limpio = []
    for h in header:
        h = h.replace(".", "")
        h = h.replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
        header_limpio.append(h)

    datos = []
    ncols = len(header_limpio)

    for ln in lineas[idx_header + 1:]:
        partes = re.split(r"\s+", ln.strip())

        # si la fila no tiene suficientes columnas, se omite
        if len(partes) < ncols:
            continue

        fila = partes[:ncols]

        # validar que primera columna parezca fecha
        fecha_txt = fila[0]
        if pd.isna(pd.to_datetime(fecha_txt, errors="coerce")):
            continue

        datos.append(fila)

    if not datos:
        return None

    df = pd.DataFrame(datos, columns=header_limpio)

    # Normalización común de columnas
    renombrar = {}

    for c in df.columns:
        c_norm = normalizar_texto(c)

        if c_norm == "FECHA":
            renombrar[c] = "fecha"
        elif "PRECIP" in c_norm or "LLUVIA" in c_norm:
            renombrar[c] = "lluvia_total_mm"
        elif "EVAP" in c_norm:
            renombrar[c] = "evaporacion_mm"
        elif c_norm in ["TMAX", "TEMPERATURA_MAXIMA", "TEMPERATURA_MAXIMA_C", "TEMPERATURA_MAXIMA_PROMEDIO"]:
            renombrar[c] = "tmax_c"
        elif c_norm in ["TMIN", "TEMPERATURA_MINIMA", "TEMPERATURA_MINIMA_C", "TEMPERATURA_MINIMA_PROMEDIO"]:
            renombrar[c] = "tmin_c"
        elif c_norm in ["TMEDIA", "TEMPERATURA_MEDIA", "TEMP_MEDIA", "TMED"]:
            renombrar[c] = "temp_media_c"

    df = df.rename(columns=renombrar)

    if "fecha" not in df.columns:
        return None

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).copy()

    # Convertir columnas numéricas conocidas
    for c in ["lluvia_total_mm", "evaporacion_mm", "tmax_c", "tmin_c", "temp_media_c"]:
        if c in df.columns:
            df[c] = df[c].apply(limpiar_numero)

    # Si no existe temp_media_c, calcularla
    if "temp_media_c" not in df.columns:
        if "tmax_c" in df.columns and "tmin_c" in df.columns:
            df["temp_media_c"] = (df["tmax_c"] + df["tmin_c"]) / 2.0
        else:
            df["temp_media_c"] = np.nan

    # Asegurar columnas mínimas
    if "lluvia_total_mm" not in df.columns:
        df["lluvia_total_mm"] = np.nan
    if "evaporacion_mm" not in df.columns:
        df["evaporacion_mm"] = np.nan
    if "tmax_c" not in df.columns:
        df["tmax_c"] = np.nan
    if "tmin_c" not in df.columns:
        df["tmin_c"] = np.nan

    df["anio"] = df["fecha"].dt.year
    df["mes_num"] = df["fecha"].dt.month
    df["dia"] = df["fecha"].dt.day
    df["season"] = df["mes_num"].apply(estacion_climatologica)

    cols_base = [
        "fecha", "anio", "mes_num", "dia",
        "lluvia_total_mm", "evaporacion_mm",
        "tmax_c", "tmin_c", "temp_media_c", "season"
    ]

    for c in cols_base:
        if c not in df.columns:
            df[c] = np.nan

    return df[cols_base].sort_values("fecha").reset_index(drop=True)


# ============================================================
# PROCESAMIENTO POR ARCHIVO
# ============================================================

def procesar_archivo_txt(ruta_txt, verbose=True):
    texto = leer_texto_robusto(ruta_txt)
    meta = obtener_metadata(texto)

    serie = parsear_tabla_diaria(texto)
    if serie is None or serie.empty:
        return None

    for k, v in meta.items():
        serie[k] = v

    serie["archivo_fuente"] = ruta_txt.name
    serie = serie.sort_values("fecha").reset_index(drop=True)

    if verbose:
        print(f"\nArchivo: {ruta_txt.name}")
        print("  Estación:", meta.get("estacion_id"), "-", meta.get("nombre"))
        print("  Fechas:", serie["fecha"].min(), "->", serie["fecha"].max())
        print("  No nulos lluvia:", serie["lluvia_total_mm"].notna().sum())
        print("  No nulos evap:", serie["evaporacion_mm"].notna().sum())
        print("  No nulos temp_media:", serie["temp_media_c"].notna().sum())
        print("  Filas:", len(serie))

    return serie


# ============================================================
# AÑOS COMPLETOS POR ESTACIÓN (DIARIO)
# ============================================================

def conservar_anios_completos(df, cobertura_min=0.90):
    """
    Conserva años con cobertura mínima en las 3 variables:
    lluvia_total_mm, evaporacion_mm, temp_media_c
    """
    d = df.dropna(subset=["lluvia_total_mm", "evaporacion_mm", "temp_media_c"]).copy()

    conteo = (
        d.groupby(["estacion_id", "anio"])["fecha"]
         .nunique()
         .reset_index(name="n_dias")
    )

    conteo["dias_esperados"] = conteo["anio"].apply(dias_esperados_en_anio)
    conteo["cobertura"] = conteo["n_dias"] / conteo["dias_esperados"]

    anios_completos = conteo[conteo["cobertura"] >= cobertura_min][["estacion_id", "anio"]]

    d = d.merge(anios_completos, on=["estacion_id", "anio"], how="inner")
    return d.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)


def resumen_estaciones(df):
    return (
        df.groupby(["estacion_id", "nombre"], dropna=False)
          .agg(
              anio_ini=("anio", "min"),
              anio_fin=("anio", "max"),
              n_anios=("anio", "nunique"),
              n_dias=("fecha", "count")
          )
          .reset_index()
          .sort_values(["n_anios", "n_dias"], ascending=False)
    )


def filtrar_estaciones_min_anios(df, min_anios=20):
    conteo = (
        df.groupby("estacion_id")["anio"]
          .nunique()
          .reset_index(name="n_anios")
    )
    validas = conteo.loc[conteo["n_anios"] >= min_anios, "estacion_id"]
    return df[df["estacion_id"].isin(validas)].copy()


# ============================================================
# PERIODO GLOBAL COMÚN
# ============================================================

def base_periodo_global(df, anio_ini=1980, anio_fin=2020):
    d = df[(df["anio"] >= anio_ini) & (df["anio"] <= anio_fin)].copy()

    total_anios_esperados = anio_fin - anio_ini + 1

    conteo = (
        d.groupby("estacion_id")["anio"]
         .nunique()
         .reset_index(name="n_anios_global")
    )
    validas = conteo.loc[conteo["n_anios_global"] == total_anios_esperados, "estacion_id"]

    d = d[d["estacion_id"].isin(validas)].copy()
    return d


# ============================================================
# DISCRETIZACIÓN DEL ESPACIO DE ESTADOS
# ============================================================

def discretizar_por_estacion(df, n_bins=2):
    out = []

    for est, g in df.groupby("estacion_id"):
        h = g.copy()

        for col, pref in [
            ("lluvia_total_mm", "P"),
            ("evaporacion_mm", "E"),
            ("temp_media_c", "T")
        ]:
            try:
                h[f"{pref}_bin"] = pd.qcut(
                    h[col],
                    q=n_bins,
                    labels=False,
                    duplicates="drop"
                ) + 1
            except ValueError:
                h[f"{pref}_bin"] = 1

        h["estado_markov"] = (
            "P" + h["P_bin"].astype(str) +
            "_E" + h["E_bin"].astype(str) +
            "_T" + h["T_bin"].astype(str)
        )

        out.append(h)

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


# ============================================================
# BASE ESTACIONAL
# ============================================================

def construir_base_estacional(df):
    d = df.copy()
    d["season_year"] = d["anio"]

    # Diciembre pertenece al invierno del año siguiente
    d.loc[d["mes_num"] == 12, "season_year"] = d.loc[d["mes_num"] == 12, "anio"] + 1

    return d.sort_values(["estacion_id", "fecha"]).reset_index(drop=True)


# ============================================================
# TRANSICIONES
# ============================================================

def transiciones_dentro_de_grupo(df, group_cols):
    d = df.sort_values(group_cols + ["fecha"]).copy()
    d["estado_siguiente"] = d.groupby(group_cols)["estado_markov"].shift(-1)
    d["fecha_siguiente"] = d.groupby(group_cols)["fecha"].shift(-1)

    d = d.dropna(subset=["estado_markov", "estado_siguiente"]).copy()
    return d


def matriz_transicion(df_trans, group_cols):
    conteos = (
        df_trans.groupby(group_cols + ["estado_markov", "estado_siguiente"])
               .size()
               .reset_index(name="n")
    )

    conteos["total_origen"] = conteos.groupby(group_cols + ["estado_markov"])["n"].transform("sum")
    conteos["prob"] = conteos["n"] / conteos["total_origen"]
    return conteos

def discretizar_precipitacion_mixta(
    serie_precip,
    n_bins_lluvia=2,
    labels_seco_lluvia=None
):
    """
    Discretiza precipitación diaria respetando la masa en cero.

    Esquema:
      - 0 : sin lluvia
      - 1..n_bins_lluvia : niveles de lluvia positiva, usando cuantiles
        SOLO sobre valores > 0

    Parámetros
    ----------
    serie_precip : pd.Series
        Serie de precipitación en mm.
    n_bins_lluvia : int
        Número de bins para los valores positivos.
        Ejemplo:
          2 -> estados finales {0,1,2}
          3 -> estados finales {0,1,2,3}
    labels_seco_lluvia : list o None
        Etiquetas opcionales para los bins positivos.
        Si None, usa enteros.

    Regresa
    -------
    pd.Series
        Serie discreta.
    """
    s = pd.to_numeric(serie_precip, errors="coerce").copy()
    out = pd.Series(index=s.index, dtype="object")

    # Estado seco
    out[s == 0] = 0

    # Valores positivos
    mask = s > 0
    s_pos = s[mask]

    if len(s_pos) == 0:
        return out.astype("Int64")

    if labels_seco_lluvia is None:
        labels = list(range(1, n_bins_lluvia + 1))
    else:
        labels = labels_seco_lluvia

    # qcut solo sobre lluvia positiva
    bins_pos = pd.qcut(
        s_pos,
        q=n_bins_lluvia,
        labels=labels,
        duplicates="drop"
    )

    out.loc[mask] = bins_pos.astype(object)

    return out.astype("Int64")
# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    print("Carpeta:", CARPETA_TXT)
    print("Existe carpeta:", CARPETA_TXT.exists())

    archivos = sorted(CARPETA_TXT.glob("*.txt"))
    print("TXT encontrados:", [a.name for a in archivos])

    if not archivos:
        print("No se encontraron .txt")
        return

    bases = []

    for ruta in archivos:
        try:
            serie = procesar_archivo_txt(ruta, verbose=True)
            if serie is not None and not serie.empty:
                bases.append(serie)
                print(f"  -> SERIE AÑADIDA: {ruta.name}")
            else:
                print(f"  -> SIN SERIE ÚTIL: {ruta.name}")
        except Exception as e:
            print(f"ERROR en {ruta.name}: {e}")

    print("\nNúmero de series construidas:", len(bases))

    if not bases:
        print("No se pudo construir ninguna serie.")
        return

    base_total = pd.concat(bases, ignore_index=True)
    print("\nTamaño base_total:", base_total.shape)

    # 1) BASE POR ESTACIÓN CON AÑOS COMPLETOS
    base_completa = conservar_anios_completos(base_total, cobertura_min=COBERTURA_MIN_ANUAL)
    print("Tamaño base_completa:", base_completa.shape)

    resumen_completos = (
        base_completa.groupby(["estacion_id", "nombre"])["anio"]
        .nunique()
        .reset_index(name="n_anios_completos")
        .sort_values("n_anios_completos", ascending=False)
    )
    print("\nTop estaciones con más años completos:")
    print(resumen_completos.head(20))

    # 2) SOLO ESTACIONES CON AL MENOS MIN_ANIOS_COMPLETOS
    base_min = filtrar_estaciones_min_anios(base_completa, min_anios=MIN_ANIOS_COMPLETOS)
    print(f"\nTamaño base_{MIN_ANIOS_COMPLETOS}:", base_min.shape)

    if base_min.empty:
        print(f"\nNo hay estaciones con al menos {MIN_ANIOS_COMPLETOS} años completos.")
        print("Prueba bajar MIN_ANIOS_COMPLETOS o COBERTURA_MIN_ANUAL.")
        return

    resumen = resumen_estaciones(base_min)

    # 3) IMPONER PERIODO GLOBAL
    base_global = base_periodo_global(base_min, anio_ini=ANIO_GLOBAL_INI, anio_fin=ANIO_GLOBAL_FIN)
    print("Tamaño base_global:", base_global.shape)

    # 4) DISCRETIZACIÓN
    base_min_disc = discretizar_por_estacion(base_min, n_bins=N_BINS)
    base_global_disc = discretizar_por_estacion(base_global, n_bins=N_BINS)

    print(f"Tamaño base_{MIN_ANIOS_COMPLETOS}_disc:", base_min_disc.shape)
    print("Tamaño base_global_disc:", base_global_disc.shape)

    # 5) BASE ESTACIONAL
    base_estacional = construir_base_estacional(base_min_disc)
    print("Tamaño base_estacional:", base_estacional.shape)

    # 6) TRANSICIONES
    trans_estacion = transiciones_dentro_de_grupo(base_min_disc, ["estacion_id"])
    mat_estacion = matriz_transicion(trans_estacion, ["estacion_id"])

    trans_season = transiciones_dentro_de_grupo(base_estacional, ["estacion_id", "season"])
    mat_season = matriz_transicion(trans_season, ["estacion_id", "season"])

    trans_mes = transiciones_dentro_de_grupo(base_min_disc, ["estacion_id", "mes_num"])
    mat_mes = matriz_transicion(trans_mes, ["estacion_id", "mes_num"])

    # 7) GUARDAR
    base_total.to_csv(SALIDA / "01_base_total_cruda.csv", index=False, encoding="utf-8-sig")
    base_completa.to_csv(SALIDA / "02_base_anios_completos.csv", index=False, encoding="utf-8-sig")
    base_min.to_csv(SALIDA / f"03_base_estaciones_{MIN_ANIOS_COMPLETOS}_anios.csv", index=False, encoding="utf-8-sig")
    resumen.to_csv(SALIDA / f"04_resumen_estaciones_{MIN_ANIOS_COMPLETOS}_anios.csv", index=False, encoding="utf-8-sig")
    base_global.to_csv(SALIDA / "05_base_global_1980_2020.csv", index=False, encoding="utf-8-sig")

    base_min_disc.to_csv(SALIDA / f"06_base_{MIN_ANIOS_COMPLETOS}_anios_discretizada.csv", index=False, encoding="utf-8-sig")
    base_global_disc.to_csv(SALIDA / "07_base_global_1980_2020_discretizada.csv", index=False, encoding="utf-8-sig")
    base_estacional.to_csv(SALIDA / "08_base_estacional.csv", index=False, encoding="utf-8-sig")

    trans_estacion.to_csv(SALIDA / "09_transiciones_por_estacion.csv", index=False, encoding="utf-8-sig")
    mat_estacion.to_csv(SALIDA / "10_matriz_transicion_por_estacion.csv", index=False, encoding="utf-8-sig")

    trans_season.to_csv(SALIDA / "11_transiciones_por_estacion_del_anio.csv", index=False, encoding="utf-8-sig")
    mat_season.to_csv(SALIDA / "12_matriz_transicion_por_estacion_del_anio.csv", index=False, encoding="utf-8-sig")

    trans_mes.to_csv(SALIDA / "13_transiciones_por_mes.csv", index=False, encoding="utf-8-sig")
    mat_mes.to_csv(SALIDA / "14_matriz_transicion_por_mes.csv", index=False, encoding="utf-8-sig")

    print("\nArchivos guardados en:", SALIDA)
    print("\nResumen de estaciones válidas:")
    print(resumen.head(20))


if __name__ == "__main__":
    main()