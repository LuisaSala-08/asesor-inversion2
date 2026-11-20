"""
Script académico de actualización de datos bursátiles de Colombia para
BOHADA & SALAZAR Asesores de Inversión (UNIPAZ — Barrancabermeja, Colombia).

Descarga históricos de 3 años desde Yahoo Finance mediante yfinance y pandas,
calcula rentabilidades, volatilidad, caída máxima, rendimiento por dividendo,
serie histórica mensual y TRM (USDCOP=X), y exporta el resultado a datos.json.
Diseñado para ejecutarse de forma robusta (siempre termina con código 0).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# Catálogo de acciones colombianas solicitadas:
# (ticker_yahoo, nombre, nemotecnico_bvc, sector)
ACCIONES_COLOMBIA: list[tuple[str, str, str, str]] = [
    ("ECOPETROL.CL", "Ecopetrol", "ECOPETROL", "Petróleo y gas"),
    ("PFCIBEST.CL", "Grupo Cibest preferencial", "PFCIBEST", "Financiero"),
    ("CIBEST.CL", "Grupo Cibest ordinaria", "CIBEST", "Financiero"),
    ("ISA.CL", "ISA", "ISA", "Energía e infraestructura"),
    ("EXITO.CL", "Grupo Éxito", "EXITO", "Comercio"),
    ("PFAVAL.CL", "Grupo Aval preferencial", "PFAVAL", "Financiero"),
    ("NUTRESA.CL", "Grupo Nutresa", "NUTRESA", "Alimentos Procesados"),
    ("GRUPOARGOS.CL", "Grupo Argos", "GRUPOARGOS", "Infraestructura"),
]

FUENTE_OFICIAL = "Yahoo Finance (yfinance); rentabilidades con precio ajustado por dividendos"
ARCHIVO_SALIDA = Path("datos.json")
TICKER_TRM = "USDCOP=X"


def redondear_pct(valor: float) -> float:
    """Convierte una proporción decimal (ej. 0.125) a porcentaje redondeado a 2 decimales (12.5)."""
    return round(float(valor) * 100.0, 2)


def normalizar_indice_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el índice de un DataFrame de yfinance a fechas sin zona horaria y ordenadas."""
    copia = df.copy()
    if copia.empty:
        return copia

    # Si yfinance devuelve MultiIndex en columnas (por ejemplo en algunas versiones), aplanarlo
    if isinstance(copia.columns, pd.MultiIndex):
        copia.columns = [col[0] if isinstance(col, tuple) else col for col in copia.columns]

    idx = pd.to_datetime(copia.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/Bogota").tz_localize(None)
    copia.index = idx
    copia = copia.sort_index()
    # Eliminar filas donde Close sea nulo o no positivo
    if "Close" in copia.columns:
        copia = copia.dropna(subset=["Close"])
        copia = copia[copia["Close"] > 0]
    return copia


def obtener_trm() -> dict[str, Any]:
    """
    Intenta descargar el tipo de cambio USDCOP=X desde Yahoo Finance.
    Si falla o no trae datos válidos, retorna {"valor": None, "fecha": None}.
    """
    try:
        trm_ticker = yf.Ticker(TICKER_TRM)
        hist = trm_ticker.history(period="5d", auto_adjust=False)
        hist = normalizar_indice_fechas(hist)
        if hist.empty or "Close" not in hist.columns:
            return {"valor": None, "fecha": None}

        ultimo_cierre = float(hist["Close"].iloc[-1])
        ultima_fecha = hist.index[-1].strftime("%Y-%m-%d")
        return {
            "valor": round(ultimo_cierre, 2),
            "fecha": ultima_fecha,
        }
    except Exception:
        return {"valor": None, "fecha": None}


def calcular_metricas_accion(
    ticker: str,
    nombre: str,
    nemotecnico: str,
    sector: str,
) -> dict[str, Any]:
    """
    Descarga 3 años de precios diarios para `ticker` y calcula todas las
    métricas requeridas. Lanza ValueError si los datos son insuficientes o inválidos.
    """
    activo = yf.Ticker(ticker)
    hist_raw = activo.history(period="3y", auto_adjust=False)
    if hist_raw is None or hist_raw.empty:
        raise ValueError("Sin datos históricos devueltos por Yahoo Finance")

    df = normalizar_indice_fechas(hist_raw)
    if len(df) < 2:
        raise ValueError("Insuficientes registros históricos (menos de 2 días)")

    if "Close" not in df.columns:
        raise ValueError("Falta la columna 'Close'")

    # Usar Adj Close si existe y tiene valores válidos; de lo contrario, respaldar en Close
    if "Adj Close" in df.columns and df["Adj Close"].notna().sum() >= 2:
        adj_close = df["Adj Close"].astype(float).ffill().bfill()
    else:
        adj_close = df["Close"].astype(float)

    close = df["Close"].astype(float)
    dividends = (
        df["Dividends"].astype(float).fillna(0.0)
        if "Dividends" in df.columns
        else pd.Series(0.0, index=df.index)
    )

    # 1. Precio actual y fecha del último cierre
    ultimo_close = float(close.iloc[-1])
    penultimo_close = float(close.iloc[-2])
    fecha_ultimo_cierre = df.index[-1].strftime("%Y-%m-%d")

    if ultimo_close <= 0 or penultimo_close <= 0:
        raise ValueError("Precios de cierre no positivos")

    # 2. Variación porcentual del día
    variacion_dia = (ultimo_close / penultimo_close) - 1.0

    # Fecha límite de 1 año atrás respecto al último dato disponible
    fecha_fin = df.index[-1]
    fecha_hace_1a = fecha_fin - pd.Timedelta(days=365)

    # Subserie del último año (desde hace 365 días hasta el último cierre)
    mascara_1a = df.index >= fecha_hace_1a
    adj_1a = adj_close.loc[mascara_1a]
    if adj_1a.empty:
        adj_1a = adj_close.tail(min(252, len(adj_close)))

    # 3. Rentabilidad a 1 año (Adj Close último / Adj Close de hace 1 año - 1)
    pos_mas_cercana_1a = int(np.argmin(np.abs((df.index - fecha_hace_1a).days)))
    adj_hace_1a = float(adj_close.iloc[pos_mas_cercana_1a])
    adj_ultimo = float(adj_close.iloc[-1])
    if adj_hace_1a <= 0:
        raise ValueError("Precio ajustado de hace 1 año inválido")
    rent_1a = (adj_ultimo / adj_hace_1a) - 1.0

    # 4. Rentabilidad anualizada en 3 años con fechas reales:
    # (Adj Close último / Adj Close primero)^(1 / años transcurridos) - 1
    adj_primero = float(adj_close.iloc[0])
    dias_transcurridos = (df.index[-1] - df.index[0]).days
    anios_transcurridos = dias_transcurridos / 365.25
    if adj_primero <= 0 or anios_transcurridos <= 0:
        raise ValueError("Plazo o precio inicial inválido para rentabilidad anualizada")
    rent_anual_3a = ((adj_ultimo / adj_primero) ** (1.0 / anios_transcurridos)) - 1.0

    # 5. Volatilidad anualizada:
    # Desviación estándar de rendimientos logarítmicos diarios del último año (Adj Close) * sqrt(252)
    ret_log_1a = np.log(adj_1a / adj_1a.shift(1)).dropna()
    if len(ret_log_1a) < 2:
        vol_anual = 0.0
    else:
        vol_anual = float(ret_log_1a.std(ddof=1)) * np.sqrt(252.0)

    # 6. Caída máxima en 1 año (mayor caída desde un máximo previo en el último año, negativa)
    maximo_acumulado_1a = adj_1a.cummax()
    drawdown_serie_1a = (adj_1a / maximo_acumulado_1a) - 1.0
    caida_maxima_1a = float(drawdown_serie_1a.min())
    # Garantizar signo negativo o cero
    if caida_maxima_1a > 0:
        caida_maxima_1a = 0.0

    # 7. Rendimiento por dividendo: suma de Dividends de los últimos 365 días / precio actual
    dividendos_1a = float(dividends.loc[mascara_1a].sum())
    dividendo_rend = dividendos_1a / ultimo_close

    # 8. Histórico mensual: último Close de cada mes de los 3 años [{"mes": "AAAA-MM", "cierre": n}]
    cierres_mensuales = close.resample("ME").last().dropna()
    historico: list[dict[str, Any]] = []
    for fecha_mes, valor_cierre in cierres_mensuales.items():
        historico.append({
            "mes": fecha_mes.strftime("%Y-%m"),
            "cierre": round(float(valor_cierre), 2),
        })

    return {
        "ticker": ticker,
        "nombre": nombre,
        "nemotecnico": nemotecnico,
        "sector": sector,
        "moneda": "COP",
        "precio": round(ultimo_close, 2),
        "fecha": fecha_ultimo_cierre,
        "variacion_dia_pct": redondear_pct(variacion_dia),
        "rent_1a_pct": redondear_pct(rent_1a),
        "rent_anual_3a_pct": redondear_pct(rent_anual_3a),
        "volatilidad_anual_pct": redondear_pct(vol_anual),
        "caida_maxima_1a_pct": redondear_pct(caida_maxima_1a),
        "dividendo_rend_pct": redondear_pct(dividendo_rend),
        "historico": historico,
    }


def main() -> int:
    """Ejecuta el proceso completo de actualización y guarda datos.json de forma segura."""
    acciones_actualizadas: list[dict[str, Any]] = []
    errores: list[dict[str, str]] = []

    for ticker, nombre, nemotecnico, sector in ACCIONES_COLOMBIA:
        try:
            registro = calcular_metricas_accion(ticker, nombre, nemotecnico, sector)
            acciones_actualizadas.append(registro)
        except Exception as exc:
            motivo = str(exc).strip() or exc.__class__.__name__
            errores.append({
                "ticker": ticker,
                "motivo": motivo,
            })

    # Si todas las acciones fallaron, conservar el archivo datos.json anterior (si existe)
    if len(acciones_actualizadas) == 0:
        print(
            "AVISO: Todas las acciones fallaron al descargar o procesar datos. "
            "No se sobrescribirá 'datos.json' para conservar la información anterior."
        )
        for err in errores:
            print(f"  - Error en {err['ticker']}: {err['motivo']}")
        print(
            f"Resumen: 0 acciones actualizadas, {len(errores)} fallaron: "
            f"{[e['ticker'] for e in errores]}"
        )
        return 0

    trm_data = obtener_trm()
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload: dict[str, Any] = {
        "actualizado": timestamp_utc,
        "fuente": FUENTE_OFICIAL,
        "trm": trm_data,
        "acciones": acciones_actualizadas,
        "errores": errores,
    }

    try:
        with ARCHIVO_SALIDA.open("w", encoding="utf-8") as archivo_json:
            json.dump(payload, archivo_json, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"AVISO: No fue posible escribir '{ARCHIVO_SALIDA}': {exc}")
        return 0

    fallidas_tickers = [e["ticker"] for e in errores]
    print(
        f"Resumen: {len(acciones_actualizadas)} acciones actualizadas, "
        f"{len(errores)} fallaron."
    )
    if fallidas_tickers:
        print(f"Acciones con error: {', '.join(fallidas_tickers)}")
    else:
        print("Acciones con error: ninguna.")

    return 0


if __name__ == "__main__":
    try:
        codigo_salida = main()
    except Exception as exc:
        print(f"AVISO: Error inesperado controlado en ejecución: {exc}")
        codigo_salida = 0
    sys.exit(codigo_salida)
