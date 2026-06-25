# -*- coding: utf-8 -*-
"""
Parcial - Calibracion del modelo Markov-modulado
================================================
PARTE C - Calibracion a datos reales y diagnostico
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute, brentq
from scipy.stats import norm

# Importamos los motores de las partes anteriores
from ParteA import price_analytic_markov
from ParteB import cos_method

# ----------------------------------------------------------------------------
# 1. Utilidades de Black-Scholes para la calibracion
# ----------------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqrtT
    d2 = d1 - sqrtT
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, option_type="call"):
    disc = np.exp(-r * T)
    lo = max(S - K * disc, 0.0) if option_type == "call" else max(K * disc - S, 0.0)
    hi = S if option_type == "call" else K * disc
    if price <= lo + 1e-6 or price >= hi - 1e-6:
        return np.nan
    def f(sig):
        return bs_price(S, K, T, r, sig, option_type) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except ValueError:
        return np.nan

# ----------------------------------------------------------------------------
# 2. Funcion Objetivo
# ----------------------------------------------------------------------------
def objective_function(theta, df, r, weight_scheme='vega'):
    sig0, sig1, lam0, lam1 = theta
    
    if sig0 <= 0 or sig1 <= 0 or lam0 <= 0 or lam1 <= 0 or sig0 >= sig1:
        return 1e6
        
    pi0 = lam1 / (lam0 + lam1)
    errors, weights = [], []
    
    for T, g in df.groupby("T"):
        K_arr = g["K"].values
        S_adj = g["S_adj"].iloc[0] 
        
        C0 = cos_method(S_adj, K_arr, T, r, sig0, sig1, lam0, lam1, regime=0, option_type="call")
        C1 = cos_method(S_adj, K_arr, T, r, sig0, sig1, lam0, lam1, regime=1, option_type="call")
        C_mod = pi0 * C0 + (1 - pi0) * C1
        
        for i, (_, row) in enumerate(g.iterrows()):
            iv_mod = implied_vol(C_mod[i], S_adj, row["K"], T, r, "call")
            if not np.isnan(iv_mod):
                errors.append(iv_mod - row["iv_mkt"])
                w = row["vega"] if weight_scheme == 'vega' else 1.0
                weights.append(w)
                
    if not errors: return 1e6
    errors, weights = np.array(errors), np.array(weights)
    weights /= np.sum(weights) 
    
    mse = np.sum(weights * errors**2)
    return np.sqrt(mse)

# ----------------------------------------------------------------------------
# 3. Pipeline de Optimizacion
# ----------------------------------------------------------------------------
def calibrate_model(df, r, weight_scheme='vega'):
    print(f"\nIniciando calibracion (Pesos: {weight_scheme})...")
    
    ranges = (slice(0.1, 0.3, 0.1), slice(0.3, 0.6, 0.1), 
              slice(0.5, 3.0, 1.0), slice(0.5, 3.0, 1.0))
              
    print(" -> Ejecutando busqueda en malla (brute)...")
    res_brute = brute(objective_function, ranges, args=(df, r, weight_scheme), finish=None, full_output=True)
    theta_0, rmse_brute = res_brute[0], res_brute[1]
    
    bounds = [(0.01, 1.0), (0.01, 1.5), (0.01, 10.0), (0.01, 10.0)]
    
    print(" -> Ejecutando Nelder-Mead...")
    res_nm = minimize(objective_function, theta_0, args=(df, r, weight_scheme),
                      method='Nelder-Mead', bounds=bounds, options={'maxiter': 300})
    
    print(" -> Ejecutando L-BFGS-B (Gradiente)...")
    res_bfgs = minimize(objective_function, theta_0, args=(df, r, weight_scheme),
                        method='L-BFGS-B', bounds=bounds)
                        
    results = {
        'Brute': {'theta': theta_0, 'rmse': rmse_brute},
        'Nelder-Mead': {'theta': res_nm.x, 'rmse': res_nm.fun},
        'L-BFGS-B': {'theta': res_bfgs.x, 'rmse': res_bfgs.fun}
    }
    
    print("\n--- RESULTADOS DE OPTIMIZACION ---")
    best_method = min(results, key=lambda k: results[k]['rmse'])
    for m, d in results.items():
        marcador = "<-- MEJOR" if m == best_method else ""
        print(f"{m:>12}: RMSE = {d['rmse']:.6f} | Theta = {np.round(d['theta'], 4)} {marcador}")
        
    return results

# ----------------------------------------------------------------------------
# 4. Graficos y Diagnosticos
# ----------------------------------------------------------------------------
def plot_convex_diagnostic(theta1, theta2, df, r, weight_scheme='vega', filename="diagnostico.png", title=""):
    print(f"\nGenerando diagnostico: {title}...")
    alphas = np.linspace(-0.5, 1.5, 40)
    valid_alphas, rmses = [], []
    
    for a in alphas:
        theta_a = a * theta1 + (1 - a) * theta2
        sig0, sig1, lam0, lam1 = theta_a
        
        if sig0 <= 0 or sig1 <= 0 or lam0 <= 0 or lam1 <= 0 or sig0 >= sig1:
            continue
            
        rmse = objective_function(theta_a, df, r, weight_scheme)
        if rmse < 1e5:
            valid_alphas.append(a)
            rmses.append(rmse)
            
    plt.figure(figsize=(8, 5))
    plt.plot(valid_alphas, rmses, 'b-', label=r'RMSE($\alpha$)')
    plt.axvline(1, color='g', linestyle='--', label='Theta 1')
    plt.axvline(0, color='r', linestyle='--', label='Theta 2')
    plt.title(f"Paisaje de Error: {title}")
    plt.xlabel("$\\alpha$ (0 = Theta 2, 1 = Theta 1)")
    plt.ylabel("RMSE (Volatilidad Implicita)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, filename)
    plt.savefig(out_path)
    print(f" -> Grafico guardado como '{filename}'")

def plot_volatility_smile(best_theta, df, r):
    print("\nGenerando graficas de la Sonrisa de Volatilidad (Modelo vs Mercado)...")
    sig0, sig1, lam0, lam1 = best_theta
    pi0 = lam1 / (lam0 + lam1)
    
    T_vals = np.sort(df["T"].unique())
    n_plots = len(T_vals)
    cols = 3
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if n_plots == 1: axes = [axes]
    elif rows == 1: axes = axes.flatten()
    else: axes = axes.flatten()
    
    for i, T in enumerate(T_vals):
        ax = axes[i]
        g = df[df["T"] == T].sort_values("K")
        K_arr = g["K"].values
        S_adj = g["S_adj"].iloc[0]
        
        C0 = cos_method(S_adj, K_arr, T, r, sig0, sig1, lam0, lam1, regime=0, option_type="call")
        C1 = cos_method(S_adj, K_arr, T, r, sig0, sig1, lam0, lam1, regime=1, option_type="call")
        C_mod = pi0 * C0 + (1 - pi0) * C1
        
        iv_mod = [implied_vol(C_mod[j], S_adj, K_arr[j], T, r, "call") for j in range(len(K_arr))]
        
        ax.plot(K_arr, g["iv_mkt"], 'ko', label="Mercado (Datos)")
        ax.plot(K_arr, iv_mod, 'b-', linewidth=2, label="Modelo Markov-Modulado")
        
        days = int(g["days"].iloc[0])
        ax.set_title(f"Vencimiento: {days} días (T={T:.2f})")
        ax.set_xlabel("Strike (K)")
        ax.set_ylabel("Volatilidad Implícita")
        ax.grid(True, alpha=0.5)
        if i == 0: ax.legend()
            
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "sonrisa_volatilidad.png")
    plt.savefig(out_path)
    print(f" -> Grafico guardado como 'sonrisa_volatilidad.png'")

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ticker = "XLE"
    
    # Manejo robusto de rutas (Busca la carpeta Snapshot un nivel arriba)
   # Manejo robusto de rutas (Busca exactamente en la misma carpeta del script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, f"snapshot_{ticker}_clean.csv")
    json_path = os.path.join(base_dir, f"snapshot_{ticker}_meta.json")
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        meta = json.load(f)
    r = meta["r"]
    
    df = df[df["type"] == "call"].copy()
    
    # 1. Calibrar (Requerimiento de comparar ponderaciones)
    res_uniform = calibrate_model(df, r, weight_scheme='uniform')
    res_vega = calibrate_model(df, r, weight_scheme='vega')
    
    theta_brute = res_vega['Brute']['theta']
    theta_bfgs = res_vega['L-BFGS-B']['theta']
    theta_nm = res_vega['Nelder-Mead']['theta']
    
    # 2. Generar Diagnosticos (AL MENOS DOS PARES)
    plot_convex_diagnostic(theta_bfgs, theta_nm, df, r, 'vega', 
                           filename="diagnostico_BFGS_vs_NM.png", title="BFGS vs Nelder-Mead")
                           
    plot_convex_diagnostic(theta_bfgs, theta_brute, df, r, 'vega', 
                           filename="diagnostico_BFGS_vs_Brute.png", title="BFGS vs Malla (Brute)")
    
    # 3. Graficas de la Sonrisa
    print("\n--- GRAFICAS DEL MODELO ---")
    plot_volatility_smile(theta_bfgs, df, r)
    # 4. Verificacion final con Ruta A (Mezcla Analitica)
    print("\n--- VERIFICACION FINAL (RUTA A) ---")
    sig0, sig1, lam0, lam1 = theta_bfgs
    pi0 = lam1 / (lam0 + lam1)
    
    # Tomamos un contrato de la mitad de la tabla para probar
    test_row = df.iloc[len(df)//2]
    S_adj, K, T = test_row["S_adj"], test_row["K"], test_row["T"]
    
    # Precio usando la formula analitica (Ruta A)
    c0_A = price_analytic_markov(S_adj, K, T, r, r, sig0, sig1, lam0, lam1, regime=0, option_type="call")
    c1_A = price_analytic_markov(S_adj, K, T, r, r, sig0, sig1, lam0, lam1, regime=1, option_type="call")
    c_mod_A = pi0 * c0_A + (1 - pi0) * c1_A
    
    # Precio usando COS (Ruta B)
    c0_B = cos_method(S_adj, K, T, r, sig0, sig1, lam0, lam1, regime=0, option_type="call")[0]
    c1_B = cos_method(S_adj, K, T, r, sig0, sig1, lam0, lam1, regime=1, option_type="call")[0]
    c_mod_B = pi0 * c0_B + (1 - pi0) * c1_B
    
    print(f"Opcion seleccionada: K={K}, T={T:.4f}")
    print(f"Precio Mezcla Analitica (Ruta A): {c_mod_A:.6f}")
    print(f"Precio Motor COS (Ruta B):        {c_mod_B:.6f}")
    print(f"Diferencia Absoluta:              {abs(c_mod_A - c_mod_B):.2e}")
