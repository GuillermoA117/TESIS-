import numpy as np
import pandas as pd


# ============================================================
# UTILIDADES BÁSICAS
# ============================================================

def _safe_log(x, base=2):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    if base == 2:
        out[mask] = np.log2(x[mask])
    else:
        out[mask] = np.log(x[mask]) / np.log(base)
    return out


def entropy_empirica(df, cols, base=2):
    """
    Entropía empírica H(X) o H(X1,...,Xk) para variables discretas.
    cols: str o lista de columnas
    """
    if isinstance(cols, str):
        cols = [cols]

    d = df.dropna(subset=cols).copy()
    n = len(d)
    if n == 0:
        return np.nan

    probs = d.groupby(cols).size().to_numpy(dtype=float) / n
    return -np.sum(probs * _safe_log(probs, base=base))


def conditional_entropy_empirica(df, target_col, given_cols, base=2):
    """
    H(target | given) para variables discretas.
    """
    if isinstance(given_cols, str):
        given_cols = [given_cols]

    cols = [target_col] + given_cols
    d = df.dropna(subset=cols).copy()
    n = len(d)
    if n == 0:
        return np.nan

    if len(given_cols) == 0:
        probs = d.groupby(target_col).size().to_numpy(dtype=float) / n
        return -np.sum(probs * _safe_log(probs, base=base))

    joint = (
        d.groupby(given_cols + [target_col])
         .size()
         .reset_index(name="n_joint")
    )

    marg = (
        d.groupby(given_cols)
         .size()
         .reset_index(name="n_given")
    )

    joint = joint.merge(marg, on=given_cols, how="left")

    p_joint = joint["n_joint"].to_numpy(dtype=float) / n
    p_cond = joint["n_joint"].to_numpy(dtype=float) / joint["n_given"].to_numpy(dtype=float)

    return -np.sum(p_joint * _safe_log(p_cond, base=base))


def mutual_information_condicional(df, y_col, x_col, z_cols=None, base=2):
    """
    I(Y ; X | Z) = H(Y|Z) - H(Y|X,Z)
    """
    if z_cols is None:
        z_cols = []
    if isinstance(z_cols, str):
        z_cols = [z_cols]

    h1 = conditional_entropy_empirica(df, target_col=y_col, given_cols=z_cols, base=base)
    h2 = conditional_entropy_empirica(df, target_col=y_col, given_cols=[x_col] + z_cols, base=base)

    if pd.isna(h1) or pd.isna(h2):
        return np.nan

    return h1 - h2


# ============================================================
# CONSTRUCCIÓN DE LAGS
# ============================================================

#def construir_lags(df, value_cols, group_cols, time_col="fecha"):
    """
    Ordena por grupo + tiempo y crea columnas *_next usando shift(-1)
    dentro de cada grupo.

    OJO:
    - Si group_cols = ["estacion_id", "mes_num"], entonces target_next
      significa 'mismo mes en el siguiente año disponible'.
    - Si group_cols = ["estacion_id"], entonces target_next significa
      'siguiente observación cronológica'.
    """
    """
    d = df.sort_values(group_cols + [time_col]).copy()

    for col in value_cols:
        d[f"{col}_next"] = d.groupby(group_cols)[col].shift(-1)

    return d
    """
def construir_lags_cronologicos(df, value_cols, group_cols=("estacion_id",), time_col="fecha"):
    """
    Construye col_next usando únicamente la siguiente observación
    si está exactamente a 1 día de distancia dentro de cada grupo.

    Si hay huecos en fechas (> 1 día), col_next se deja como NaN.
    """
    d = df.sort_values(list(group_cols) + [time_col]).copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    # Fecha siguiente dentro del grupo
    d["__fecha_next"] = d.groupby(list(group_cols))[time_col].shift(-1)

    # Diferencia en días hacia la siguiente observación
    d["__delta_dias_next"] = (d["__fecha_next"] - d[time_col]).dt.days

    # Construir next solo si el salto es exactamente 1 día
    for col in value_cols:
        siguiente = d.groupby(list(group_cols))[col].shift(-1)
        d[f"{col}_next"] = siguiente.where(d["__delta_dias_next"] == 1)

    # Limpiar auxiliares
    d = d.drop(columns=["__fecha_next", "__delta_dias_next"])

    return d
# ============================================================
# 1) ENTROPÍA CONDICIONAL DEL ESTADO MARKOVIANO
# ============================================================

def entropia_condicional_por_grupo(
    df,
    state_col="estado_markov",
    group_cols=None,
    time_col="fecha",
    base=2
):
    """
    Calcula H(X_{t+1} | X_t) dentro de cada grupo indicado.

    Ejemplos:
    - group_cols=["estacion_id"]              -> global por estación
    - group_cols=["estacion_id","mes_num"]    -> mensual por estación
    - group_cols=["estacion_id","season"]     -> estacional por estación
    """
    if group_cols is None:
        group_cols = ["estacion_id"]

    d = construir_lags_cronologicos(df, value_cols=[state_col], group_cols=group_cols, time_col=time_col)
    d = d.dropna(subset=[state_col, f"{state_col}_next"]).copy()

    resultados = []

    for claves, g in d.groupby(group_cols, dropna=False):
        if not isinstance(claves, tuple):
            claves = (claves,)

        n = len(g)
        h = conditional_entropy_empirica(
            g,
            target_col=f"{state_col}_next",
            given_cols=[state_col],
            base=base
        )

        fila = {col: val for col, val in zip(group_cols, claves)}
        fila["n_transiciones"] = n
        fila["H_condicional"] = h
        resultados.append(fila)

    return pd.DataFrame(resultados)


def entropia_condicional_promedio_periodica(
    df,
    period_col,
    state_col="estado_markov",
    station_col="estacion_id",
    time_col="fecha",
    base=2
):
    """
    Calcula:
    - H(X_{t+1}|X_t, period=valor) por estación y periodo
    - promedio ponderado por la frecuencia empírica del periodo

    period_col puede ser "mes_num" o "season".
    """
    group_cols = [station_col, period_col]

    h_periodo = entropia_condicional_por_grupo(
        df=df,
        state_col=state_col,
        group_cols=group_cols,
        time_col=time_col,
        base=base
    )

    # Pesos empíricos P(periodo | estación), usando número de transiciones
    pesos = (
        h_periodo.groupby(station_col)["n_transiciones"]
        .transform(lambda x: x / x.sum() if x.sum() > 0 else np.nan)
    )

    h_periodo["peso_periodo"] = pesos
    h_periodo["contribucion"] = h_periodo["peso_periodo"] * h_periodo["H_condicional"]

    h_prom = (
        h_periodo.groupby(station_col, dropna=False)
        .agg(
            H_condicional_promedio=("contribucion", "sum"),
            n_transiciones_total=("n_transiciones", "sum")
        )
        .reset_index()
    )

    return h_periodo, h_prom


# ============================================================
# 2) FLUJO DE INFORMACIÓN / TRANSFER ENTROPY
# ============================================================

def transfer_entropy_schreiber(
    df,
    source_col,
    target_col,
    group_cols=None,
    time_col="fecha",
    base=2
):
    """
    TE clásica:
        I(target_{t+1} ; source_t | target_t)

    Devuelve TE por grupo.
    """
    if group_cols is None:
        group_cols = ["estacion_id"]

    d = construir_lags_cronologicos(
        df,
        value_cols=[source_col, target_col],
        group_cols=group_cols,
        time_col=time_col
    )

    d = d.dropna(subset=[source_col, target_col, f"{target_col}_next"]).copy()

    resultados = []

    for claves, g in d.groupby(group_cols, dropna=False):
        if not isinstance(claves, tuple):
            claves = (claves,)

        te = mutual_information_condicional(
            g,
            y_col=f"{target_col}_next",
            x_col=source_col,
            z_cols=[target_col],
            base=base
        )

        h_target_next = entropy_empirica(g, f"{target_col}_next", base=base)
        h_cond_target = conditional_entropy_empirica(
            g,
            target_col=f"{target_col}_next",
            given_cols=[target_col],
            base=base
        )

        fila = {col: val for col, val in zip(group_cols, claves)}
        fila["n_muestras"] = len(g)
        fila["TE_schreiber"] = te
        fila["H_target_next"] = h_target_next
        fila["H_target_next_given_target"] = h_cond_target
        fila["TE_norm_por_Htarget"] = te / h_target_next if h_target_next and h_target_next > 0 else np.nan
        fila["TE_norm_por_Hcond"] = te / h_cond_target if h_cond_target and h_cond_target > 0 else np.nan
        resultados.append(fila)

    return pd.DataFrame(resultados)


def flujo_informacion_multivariado(
    df,
    source_col,
    target_col,
    observable_cols,
    group_cols=None,
    time_col="fecha",
    base=2
):
    """
    Flujo de información tipo observable-state del artículo:

        T^Y_{source -> target}
        = H(target_{t+1} | Y_t sin source)
          - H(target_{t+1} | Y_t completo)

    donde Y_t = observable_cols.

    Ejemplo climático:
    observable_cols = ["P_bin", "E_bin", "T_bin"]
    source_col      = "P_bin"
    target_col      = "T_bin"

    entonces:
    T_{P->T} = H(T_{t+1} | E_t, T_t) - H(T_{t+1} | P_t, E_t, T_t)
    """
    if group_cols is None:
        group_cols = ["estacion_id"]

    d = construir_lags_cronologicos(
        df,
        value_cols=list(set(observable_cols)),
        group_cols=group_cols,
        time_col=time_col
    )

    target_next = f"{target_col}_next"

    needed = list(set(observable_cols + [target_next]))
    d = d.dropna(subset=needed).copy()

    if source_col not in observable_cols or target_col not in observable_cols:
        raise ValueError("source_col y target_col deben pertenecer a observable_cols")

    cond_full = observable_cols.copy()
    cond_without_source = [c for c in observable_cols if c != source_col]

    resultados = []

    for claves, g in d.groupby(group_cols, dropna=False):
        if not isinstance(claves, tuple):
            claves = (claves,)

        h_without = conditional_entropy_empirica(
            g,
            target_col=target_next,
            given_cols=cond_without_source,
            base=base
        )

        h_full = conditional_entropy_empirica(
            g,
            target_col=target_next,
            given_cols=cond_full,
            base=base
        )

        flujo = h_without - h_full
        h_target_next = entropy_empirica(g, target_next, base=base)
        leak = conditional_entropy_empirica(
            g,
            target_col=target_next,
            given_cols=observable_cols,
            base=base
        )

        fila = {col: val for col, val in zip(group_cols, claves)}
        fila["n_muestras"] = len(g)
        fila["H_sin_fuente"] = h_without
        fila["H_con_todas"] = h_full
        fila["flujo_info"] = flujo
        fila["H_target_next"] = h_target_next
        fila["info_leak"] = leak
        fila["flujo_norm_Htarget"] = flujo / h_target_next if h_target_next and h_target_next > 0 else np.nan
        fila["leak_norm_Htarget"] = leak / h_target_next if h_target_next and h_target_next > 0 else np.nan
        resultados.append(fila)

    return pd.DataFrame(resultados)


# ============================================================
# 3) MATRIZ COMPLETA DE FLUJOS ENTRE P, E, T
# ============================================================

def matriz_flujos_observables(
    df,
    observable_cols=("P_bin", "E_bin", "T_bin"),
    group_cols=None,
    time_col="fecha",
    base=2
):
    """
    Calcula todos los flujos i -> j para i,j en observable_cols.
    """
    if group_cols is None:
        group_cols = ["estacion_id"]

    salidas = []

    for source in observable_cols:
        for target in observable_cols:
            res = flujo_informacion_multivariado(
                df=df,
                source_col=source,
                target_col=target,
                observable_cols=list(observable_cols),
                group_cols=group_cols,
                time_col=time_col,
                base=base
            ).copy()
            res["source"] = source
            res["target"] = target
            salidas.append(res)

    return pd.concat(salidas, ignore_index=True)


# ============================================================
# 4) EJEMPLO DE USO CON TUS CSV
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # CAMBIA ESTA RUTA
    # --------------------------------------------------------
    ruta = r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\salida_conagua_gto\diarios_txt\salidas_markov_diarios\06_base_15_anios_discretizada.csv"

    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # --------------------------------------------------------
    # A) ENTROPÍA CONDICIONAL GLOBAL
    #    H(X_{t+1} | X_t)
    # --------------------------------------------------------
    H_global = entropia_condicional_por_grupo(
        df=df,
        state_col="estado_markov",
        group_cols=["estacion_id"],
        time_col="fecha",
        base=2
    )
    H_global.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\H_condicional_global_por_estacion.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # B) ENTROPÍA CONDICIONAL MENSUAL
    #    H(X_{t+1} | X_t, mes=m)
    #
    # OJO: con tus grupos actuales esto significa
    #      "mismo mes entre años consecutivos disponibles"
    # --------------------------------------------------------
    H_mes, H_mes_prom = entropia_condicional_promedio_periodica(
        df=df,
        period_col="mes_num",
        state_col="estado_markov",
        station_col="estacion_id",
        time_col="fecha",
        base=2
    )
    H_mes.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\H_condicional_mensual_por_estacion_y_mes.csv", index=False, encoding="utf-8-sig")
    H_mes_prom.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\H_condicional_promedio_mensual_por_estacion.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # C) ENTROPÍA CONDICIONAL ESTACIONAL
    #    H(X_{t+1} | X_t, season=s)
    # --------------------------------------------------------
    H_season, H_season_prom = entropia_condicional_promedio_periodica(
        df=df,
        period_col="season",
        state_col="estado_markov",
        station_col="estacion_id",
        time_col="fecha",
        base=2
    )
    H_season.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\H_condicional_estacional_por_estacion_y_season.csv", index=False, encoding="utf-8-sig")
    H_season_prom.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results\H_condicional_promedio_estacional_por_estacion.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # D) TRANSFER ENTROPY CLÁSICA
    #    Ejemplo: P_t -> T_{t+1} condicionado en T_t
    # --------------------------------------------------------
    TE_PT_mes = transfer_entropy_schreiber(
        df=df,
        source_col="P_bin",
        target_col="T_bin",
        group_cols=["estacion_id", "mes_num"],
        time_col="fecha",
        base=2
    )
    TE_PT_mes.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\TE_schreiber_P_to_T_por_estacion_y_mes.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # E) FLUJO DE INFORMACIÓN MULTIVARIADO
    #    Ejemplo: P_t -> T_{t+1}, condicionado en todas
    #             las demás observables presentes
    #
    # T_{P->T} = H(T_{t+1}|E_t,T_t) - H(T_{t+1}|P_t,E_t,T_t)
    # --------------------------------------------------------
    flujo_PT_mes = flujo_informacion_multivariado(
        df=df,
        source_col="P_bin",
        target_col="T_bin",
        observable_cols=["P_bin", "E_bin", "T_bin"],
        group_cols=["estacion_id", "mes_num"],
        time_col="fecha",
        base=2
    )
    flujo_PT_mes.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\flujo_info_multivariado_P_to_T_por_estacion_y_mes.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # F) MATRIZ COMPLETA DE FLUJOS ENTRE P, E, T
    # --------------------------------------------------------
    matriz_flujos_mes = matriz_flujos_observables(
        df=df,
        observable_cols=("P_bin", "E_bin", "T_bin"),
        group_cols=["estacion_id", "mes_num"],
        time_col="fecha",
        base=2
    )
    matriz_flujos_mes.to_csv(r"C:\Users\Guillermo\Desktop\Maestria\TESIS\cods\results diarios\matriz_flujos_PET_por_estacion_y_mes.csv", index=False, encoding="utf-8-sig")

    print("Listo. Archivos generados.")