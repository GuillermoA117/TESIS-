import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

# =========================================================
# CONFIGURACION
# =========================================================

# Archivo HTML guardado
HTML_LOCAL = r"C:\Users\Guillermo\Desktop\Maestria\TESIS\Normales Climatológicas por Estado.html"

# Carpeta de salida
OUT_DIR = Path("salida_conagua_gto")
OUT_DIR.mkdir(exist_ok=True)

# Estado
ESTADO = "gto"

# Página real de recursos
BASE_RECURSOS = "https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas"

# Esto para evitar que truene por certificados
VERIFICAR_SSL = False

# Quitar warnings de HTTPS inseguro
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def get_text(url: str) -> str:
    """
    Descarga texto desde una URL.
    """
    r = requests.get(url, verify=VERIFICAR_SSL, timeout=60)
    r.raise_for_status()
    return r.text


def construir_url_mensual(clave: str, estado: str = ESTADO) -> str:
    """
    Construye la URL correcta del archivo mensual.
    """
    clave = str(clave).strip().replace(".0", "").zfill(5)
    return f"{BASE_RECURSOS}/Mensuales/{estado}/mes{clave}.txt"


def construir_url_diaria(clave: str, estado: str = ESTADO) -> str:
    """
    Construye la URL correcta del archivo diario.
    """
    clave = str(clave).strip().replace(".0", "").zfill(5)
    return f"{BASE_RECURSOS}/Diarios/{estado}/dia{clave}.txt"


def parse_conagua_txt(txt: str) -> pd.DataFrame:
    """
    Intenta parsear un TXT de CONAGUA.
    Funciona para diarios y, en varios casos, también para mensuales.
    """
    lines = txt.splitlines()

    start = None
    for i, line in enumerate(lines):
        line_up = line.upper()

        if "FECHA" in line_up:
            start = i
            break

        if ("ANIO" in line_up or "AÑO" in line_up) and "MES" in line_up:
            start = i
            break

    if start is None:
        raise ValueError("No encontré encabezado tabular ('FECHA' o 'ANIO/MES').")

    block = "\n".join(lines[start:])

    df = pd.read_csv(
        StringIO(block),
        sep=r"\s+",
        engine="python"
    )

    # Normalizar nombres de columnas
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Intentar construir FECHA
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    elif "ANIO" in df.columns and "MES" in df.columns:
        df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
        df["FECHA"] = pd.to_datetime(
            dict(year=df["ANIO"], month=df["MES"], day=1),
            errors="coerce"
        )

    else:
        raise ValueError("No pude construir la columna FECHA.")

    # Convertir resto de columnas a numérico cuando se pueda
    for c in df.columns:
        if c != "FECHA":
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df = df.dropna(subset=["FECHA"]).copy()

    return df


# =========================================================
# 1) LEER HTML LOCAL Y EXTRAER TABLA DE ESTACIONES
# =========================================================

html = Path(HTML_LOCAL).read_text(encoding="utf-8", errors="ignore")
soup = BeautifulSoup(html, "html.parser")
iframe = soup.find("iframe")
if iframe is None:
    raise RuntimeError("No encontré el iframe en el HTML guardado.")

iframe_src = iframe.get("src", "").strip()
print("iframe encontrado:", iframe_src)

catalog_url = f"https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/catalogo/cat_{ESTADO}.html"
print("Catálogo URL:", catalog_url)

catalog_html = get_text(catalog_url)
catalog_soup = BeautifulSoup(catalog_html, "html.parser")

table = catalog_soup.find("table")
if table is None:
    raise RuntimeError("No encontré la tabla dentro de cat_gto.html")

rows = table.find_all("tr")

estaciones = []

for tr in rows[1:]:
    tds = tr.find_all("td")
    if len(tds) < 4:
        continue

    clave = tds[0].get_text(strip=True)
    nombre = tds[1].get_text(strip=True)
    municipio = tds[2].get_text(strip=True)
    situacion = tds[3].get_text(strip=True)

    clave = str(clave).replace(".0", "").zfill(5)

    estaciones.append({
        "CLAVE": clave,
        "NOMBRE": nombre,
        "MUNICIPIO": municipio,
        "SITUACION": situacion,
        "URL_DIARIA": construir_url_diaria(clave, ESTADO),
        "URL_MENSUAL": construir_url_mensual(clave, ESTADO),
    })

df_estaciones = pd.DataFrame(estaciones)

print("Total estaciones:", len(df_estaciones))

df_estaciones.to_csv(OUT_DIR / "catalogo_completo.csv", index=False, encoding="utf-8")

df_operando = df_estaciones[
    df_estaciones["SITUACION"].str.strip().str.lower() == "operando"
].copy()

print("Operando:", len(df_operando))

df_operando.to_csv(OUT_DIR / "catalogo_operando.csv", index=False, encoding="utf-8")


# =========================================================
# 2) DESCARGAR DIARIOS
# =========================================================

diarios_dir = OUT_DIR / "diarios_txt"
diarios_dir.mkdir(exist_ok=True)

descargadas = []
fallas = []

for _, row in df_operando.iterrows():
    clave = row["CLAVE"]
    nombre = row["NOMBRE"]
    url = row["URL_DIARIA"]

    try:
        r = requests.get(url, verify=VERIFICAR_SSL, timeout=60)
        r.raise_for_status()

        txt = r.text

        txt_path = diarios_dir / f"dia{clave}.txt"
        txt_path.write_text(txt, encoding="utf-8", errors="ignore")

        descargadas.append({
            "CLAVE": clave,
            "NOMBRE": nombre,
            "URL_DIARIA": url,
            "ARCHIVO": str(txt_path)
        })

        print("OK diario:", clave, nombre)

        time.sleep(0.15)

    except Exception as e:
        fallas.append({
            "CLAVE": clave,
            "NOMBRE": nombre,
            "URL_DIARIA": url,
            "ERROR": str(e)
        })
        print("FALLÓ diario:", clave, nombre, e)

df_descargadas = pd.DataFrame(descargadas)
df_fallas = pd.DataFrame(fallas)

df_descargadas.to_csv(OUT_DIR / "diarios_descargados.csv", index=False, encoding="utf-8")
df_fallas.to_csv(OUT_DIR / "fallas_diario.csv", index=False, encoding="utf-8")


# =========================================================
# 3) PARSEAR TXTS DIARIOS A DATAFRAMES
# =========================================================

dfs = []
parse_fallas = []

for _, row in df_descargadas.iterrows():
    clave = row["CLAVE"]
    nombre = row["NOMBRE"]
    archivo = row["ARCHIVO"]

    try:
        txt = Path(archivo).read_text(encoding="utf-8", errors="ignore")
        dfd = parse_conagua_txt(txt)

        dfd["CLAVE"] = clave
        dfd["ESTACION"] = nombre

        municipio = df_operando.loc[df_operando["CLAVE"] == clave, "MUNICIPIO"].iloc[0]
        dfd["MUNICIPIO"] = municipio

        dfs.append(dfd)

        print("Parseado:", clave, nombre)

    except Exception as e:
        parse_fallas.append({
            "CLAVE": clave,
            "NOMBRE": nombre,
            "ARCHIVO": archivo,
            "ERROR": str(e)
        })
        print("FALLÓ parseo:", clave, nombre, e)

if len(dfs) > 0:
    df_clima_diario = pd.concat(dfs, ignore_index=True)
else:
    df_clima_diario = pd.DataFrame()

df_clima_diario.to_csv(OUT_DIR / "clima_diario_operando.csv", index=False, encoding="utf-8")
pd.DataFrame(parse_fallas).to_csv(OUT_DIR / "fallas_parseo_diario.csv", index=False, encoding="utf-8")


# =========================================================
# 4) EJEMPLOS PARA ANALISIS CLIMATICO
# =========================================================

if not df_clima_diario.empty:
    print("\nPrimeras filas del dataset final:")
    print(df_clima_diario.head())

    if "TMAX" in df_clima_diario.columns:
        df_clima_diario["ESTADO_TMAX"] = pd.cut(
            df_clima_diario["TMAX"],
            bins=[-50, 15, 25, 60],
            labels=["frio", "templado", "calido"]
        )

        df_clima_diario.to_csv(
            OUT_DIR / "clima_diario_operando_con_estados.csv",
            index=False,
            encoding="utf-8"
        )

print("\nListo.")
print("Resultados guardados en:", OUT_DIR.resolve())