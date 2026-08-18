import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def handler(event, context):
    """
    Netlify Serverless Function en Python para optimización de carteras (Markowitz y CAPM/SML).
    Recibe los parámetros del frontend vía POST (JSON), descarga datos en tiempo real de yfinance,
    ejecuta las optimizaciones matemáticas cuadráticas y retorna un JSON estructurado con resultados.
    """
    # 1. Configurar headers CORS para desarrollo y producción
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Content-Type": "application/json"
    }

    # Manejar solicitud preflight de CORS (OPTIONS)
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": headers,
            "body": ""
        }

    try:
        # Importar yfinance dentro del handler para optimizar carga en serverless
        import yfinance as yf
    except ImportError:
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "La librería yfinance no está disponible en el entorno."})
        }

    # 2. Parsear parámetros recibidos del frontend
    try:
        body = json.loads(event.get("body", "{}"))
    except Exception:
        body = {}

    # Parámetros por defecto con rigor financiero
    tickers_input = body.get("tickers", "AAPL, MSFT, GOOGL, AMZN")
    benchmark = body.get("benchmark", "SPY")
    periodo = body.get("period", "3y")  # '1y', '3y', '5y'
    
    # Procesar tickers ingresados por el usuario
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Añadir benchmark y bono del tesoro a 10 años (^TNX) como Risk-Free de mercado
    todos_tickers = list(set(tickers + [benchmark, "^TNX"]))

    # 3. Descarga de datos mediante yfinance
    try:
        data = yf.download(todos_tickers, period=periodo)["Adj Close"]
        if data.empty or len(data) < 10:
            raise ValueError("Datos insuficientes descargados de yfinance.")
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": f"Error descargando datos financieros: {str(e)}"})
        }

    # Separar activos, benchmark y risk-free
    precios_activos = data[tickers].dropna()
    precios_benchmark = data[benchmark].dropna()
    
    # Obtener la tasa libre de riesgo dinámica actual usando el cierre del bono a 10 años (^TNX)
    rf_serie = data["^TNX"].dropna()
    if not rf_serie.empty:
        # ^TNX reporta en puntos porcentuales (ej. 4.25), lo dividimos por 100 para decimalizar
        rf_anual = float(rf_serie.iloc[-1]) / 100.0
    else:
        rf_anual = 0.04  # Fallback a 4% si no hay TNX disponible

    rf_diaria = rf_anual / 252

    # Alineación de índices de fechas para retornos
    retornos_activos = precios_activos.pct_change().dropna()
    retornos_benchmark = precios_benchmark.pct_change().dropna()
    
    # Sincronizar fechas
    idx_comun = retornos_activos.index.intersection(retornos_benchmark.index)
    retornos_activos = retornos_activos.loc[idx_comun]
    retornos_benchmark = retornos_benchmark.loc[idx_comun]

    # 4. Cálculos Estadísticos Anualizados (252 días hábiles)
    mu = retornos_activos.mean() * 252  # Retornos esperados anuales
    cov_matrix = retornos_activos.cov() * 252  # Matriz de covarianza anualizada
    
    # Retorno esperado y volatilidad del benchmark (SPY)
    ret_m = float(retornos_benchmark.mean()) * 252
    var_m = float(retornos_benchmark.var()) * 252
    vol_m = np.sqrt(var_m)

    # Calcular Betas individuales frente al mercado (SPY)
    betas_individuales = {}
    for t in tickers:
        cov_i_m = retornos_activos[t].cov(retornos_benchmark) * 252
        betas_individuales[t] = float(cov_i_m / var_m)

    # Coeficientes de correlación para el frontend
    corr_df = retornos_activos.corr()
    correlaciones = {
        "tickers": tickers,
        "matrix": corr_df.values.tolist()
    }

    # 5. Funciones de Optimización Cuadrática (Markowitz)
    num_activos = len(tickers)
    bounds = tuple((0.0, 1.0) for _ in range(num_activos))  # No se permiten ventas en corto
    restriccion_suma_pesos = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # A. Portafolio de Mínima Varianza Global (MVG)
    def var_portafolio(w):
        return np.dot(w.T, np.dot(cov_matrix, w))

    peso_inicial = np.array([1.0 / num_activos] * num_activos)
    res_mvg = minimize(
        var_portafolio,
        peso_inicial,
        method="SLSQP",
        bounds=bounds,
        constraints=restriccion_suma_pesos
    )
    pesos_mvg = res_mvg.x.tolist()
    vol_mvg = np.sqrt(res_mvg.fun)
    ret_mvg = np.sum(np.array(pesos_mvg) * mu)
    beta_mvg = sum(pesos_mvg[i] * betas_individuales[tickers[i]] for i in range(num_activos))

    # B. Portafolio Riesgoso Óptimo (PRO - Máximo Sharpe)
    def sharpe_negativo(w):
        ret_p = np.sum(w * mu)
        vol_p = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return - (ret_p - rf_anual) / vol_p if vol_p > 0 else 0

    res_pro = minimize(
        sharpe_negativo,
        peso_inicial,
        method="SLSQP",
        bounds=bounds,
        constraints=restriccion_suma_pesos
    )
    pesos_pro = res_pro.x.tolist()
    vol_pro = np.sqrt(var_portafolio(res_pro.x))
    ret_pro = np.sum(np.array(pesos_pro) * mu)
    sharpe_pro = (ret_pro - rf_anual) / vol_pro
    beta_pro = sum(pesos_pro[i] * betas_individuales[tickers[i]] for i in range(num_activos))

    # C. Generación de Puntos de la Frontera Eficiente
    # Generamos un rango de retornos objetivo entre el MVG y el máximo activo individual
    max_ret_activo = mu.max()
    retornos_objetivo = np.linspace(ret_mvg, max_ret_activo, 25)
    puntos_frontera = []

    for r_obj in retornos_objetivo:
        restriccion_retorno = {"type": "eq", "fun": lambda w: np.sum(w * mu) - r_obj}
        restricciones = [restriccion_suma_pesos, restriccion_retorno]
        res_opt = minimize(
            var_portafolio,
            peso_inicial,
            method="SLSQP",
            bounds=bounds,
            constraints=restricciones
        )
        if res_opt.success:
            vol_opt = np.sqrt(res_opt.fun)
            puntos_frontera.append({"volatilidad": float(vol_opt), "retorno": float(r_obj)})

    # 6. Datos individuales de los activos
    datos_activos = []
    for i, t in enumerate(tickers):
        vol_i = np.sqrt(cov_matrix.iloc[i, i])
        datos_activos.append({
            "ticker": t,
            "retorno": float(mu.iloc[i]),
            "volatilidad": float(vol_i),
            "beta": float(betas_individuales[t]),
            "sharpe": float((mu.iloc[i] - rf_anual) / vol_i) if vol_i > 0 else 0
        })

    # 7. Compilar resultados de retorno JSON
    respuesta = {
        "rf_rate": float(rf_anual),
        "benchmark": {
            "ticker": benchmark,
            "retorno": float(ret_m),
            "volatilidad": float(vol_m),
            "beta": 1.0
        },
        "mvg": {
            "pesos": dict(zip(tickers, pesos_mvg)),
            "retorno": float(ret_mvg),
            "volatilidad": float(vol_mvg),
            "sharpe": float((ret_mvg - rf_anual) / vol_mvg),
            "beta": float(beta_mvg)
        },
        "pro": {
            "pesos": dict(zip(tickers, pesos_pro)),
            "retorno": float(ret_pro),
            "volatilidad": float(vol_pro),
            "sharpe": float(sharpe_pro),
            "beta": float(beta_pro)
        },
        "frontera_eficiente": puntos_frontera,
        "activos": datos_activos,
        "correlaciones": correlaciones
    }

    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(respuesta)
    }
