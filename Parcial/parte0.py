# -*- coding: utf-8 -*-
"""
Parcial - Calibracion del modelo Markov-modulado
================================================
PARTE 0 - Construccion del conjunto de datos (ETF asignado: XLE)

Objetivo (enunciado, Parte 0):
  1. Descargar la cadena de opciones del ETF para >= 5-6 vencimientos entre
     ~1 mes y ~1 anio, documentando fecha/hora del snapshot.
  2. Congelar la cadena en un CSV (la calibracion correra contra ESE archivo,
     no contra una descarga en vivo).
  3. Filtrar: descartar opciones sin volumen/interes abierto, con spread
     bid-ask excesivo, o muy lejos del dinero (0.8 <= K/S0 <= 1.2). Usar el mid.
  4. Fijar r (T-bills) y q (dividendos). Trabajar con el forward
     F = S0 e^{(r-q)T}, o equivalently absorber q en el spot S0 e^{-qT}.
  5. Reportar una tabla con el numero de cotizaciones por vencimiento tras filtrar.

Decision de modelado (justificada en el informe):
  Las tres rutas de valoracion (COS, formula analitica, EDP) estan escritas
  para un GBM SIN dividendos a tasa unica r. Por eso ABSORBEMOS q en el spot:
  para cada vencimiento usamos un spot efectivo S_adj(T) = F_T * e^{-rT},
  donde F_T es el forward implicito por paridad put-call observado en el
  mercado. Asi el forward del modelo iguala EXACTAMENTE al de mercado y todas
  las cotizaciones (calls y puts) se valoran con un unico r. Ademas no hace
  falta una q externa "a ojo": la inferimos del propio mercado.

Reproducibilidad:
  - Se fija una semilla (solo afecta al modo --selftest).
  - El snapshot se escribe en CSV + un sidecar JSON con metadatos
    (timestamp, S0, r, fuente). La calibracion debe leer ESE CSV.

Uso:
  python parte0_datos.py            # descarga real desde Yahoo (requiere internet)
  python parte0_datos.py --selftest # corre el pipeline sobre una cadena sintetica
                                     # (sin red) para validar filtrado/forward/IV.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------------
TICKER          = "XLE"      # <-- ETF asignado. Cambiar aqui si fuese necesario.
SEED            = 12345

# Filtros
MONEYNESS_LO    = 0.80       # K/S0 minimo
MONEYNESS_HI    = 1.20       # K/S0 maximo
MAX_REL_SPREAD  = 0.50       # (ask-bid)/mid maximo permitido
MIN_MID         = 0.05       # precio mid minimo (evita "penny options")
REQUIRE_OI      = True       # descartar interes abierto == 0

# Seleccion de vencimientos (~1 mes a ~1 anio)
MIN_DAYS        = 25
MAX_DAYS        = 380
MAX_EXPIRIES    = 12         # tope para mantener el dataset manejable

# Tasa libre de riesgo
R_OVERRIDE      = None       # p.ej. 0.0525 para fijarla a mano; None => usar ^IRX

# Salidas
OUT_RAW         = f"snapshot_{TICKER}_raw.csv"
OUT_CLEAN       = f"snapshot_{TICKER}_clean.csv"
OUT_META        = f"snapshot_{TICKER}_meta.json"

DAYS_PER_YEAR   = 365.0


# ============================================================================
# 1. UTILIDADES DE BLACK-SCHOLES (sin dividendos; q ya absorbido en S_adj)
# ============================================================================
def bs_price(S, K, T, r, sigma, option_type="call"):
    """BS europeo sin dividendos. S = spot efectivo (ya con q absorbido)."""
    S, K = float(S), float(K)
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqrtT
    d2 = d1 - sqrtT
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    """Vega de BS (identica para call y put)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqrtT
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, option_type="call"):
    """IV de BS por brentq, con chequeo de cotas de no-arbitraje."""
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    disc = np.exp(-r * T)
    if option_type == "call":
        lo, hi = max(S - K * disc, 0.0), S          # cotas de la call
    else:
        lo, hi = max(K * disc - S, 0.0), K * disc    # cotas de la put
    if price <= lo + 1e-10 or price >= hi - 1e-12:
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, sig, option_type) - price

    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10, maxiter=200)
    except ValueError:
        return np.nan


# ============================================================================
# 2. FORWARD IMPLICITO POR PARIDAD PUT-CALL (por vencimiento)
# ============================================================================
def implied_forward(df_T, r, T, S0):
    """
    Estima el forward F_T y la q implicita para un vencimiento.

    Paridad con dividendo continuo q:
        C - P = S0 e^{-qT} - K e^{-rT} = e^{-rT} (F_T - K),  F_T = S0 e^{(r-q)T}.
    Para strikes fijos: (C-P) vs K es lineal, pendiente = -e^{-rT},
    intercepto = e^{-rT} F_T. Hacemos minimos cuadrados sobre los strikes
    donde existen AMBOS (call y put) cerca del dinero.

    Devuelve (F_T, q_impl, n_pares). Si no hay suficientes pares devuelve
    el fallback F_T = S0 e^{rT} (q=0) marcado con n_pares = 0.
    """
    calls = df_T[df_T["type"] == "call"][["K", "mid"]].rename(columns={"mid": "C"})
    puts  = df_T[df_T["type"] == "put"][["K", "mid"]].rename(columns={"mid": "P"})
    m = pd.merge(calls, puts, on="K", how="inner")
    # cerca del dinero: la paridad es mas limpia donde ambos son liquidos
    m = m[(m["K"] >= MONEYNESS_LO * S0) & (m["K"] <= MONEYNESS_HI * S0)]
    if len(m) < 3:
        return S0 * np.exp(r * T), 0.0, 0

    y = (m["C"] - m["P"]).to_numpy()
    K = m["K"].to_numpy()
    # regresion y = a + b*K  =>  b ~ -e^{-rT}, a ~ e^{-rT} F_T
    A = np.vstack([np.ones_like(K), K]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    if b >= 0:                       # pendiente debe ser negativa
        return S0 * np.exp(r * T), 0.0, 0
    disc = -b                        # e^{-rT} implicito por los datos
    F_T = a / disc
    q_impl = r - np.log(F_T / S0) / T
    return float(F_T), float(q_impl), int(len(m))


# ============================================================================
# 3. DESCARGA (yfinance) -> DataFrame largo y crudo
# ============================================================================
def fetch_chain_yfinance(ticker):
    """Descarga spot, tasa y cadena. Devuelve (df_raw, meta)."""
    import yfinance as yf

    snap_ts = datetime.now(timezone.utc)
    tk = yf.Ticker(ticker)

    # --- Spot S0 ---
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No hay historico de precios para {ticker}.")
    S0 = float(hist["Close"].iloc[-1])

    # --- Tasa libre de riesgo r ---
    if R_OVERRIDE is not None:
        r = float(R_OVERRIDE)
        r_source = f"override manual = {r:.4%}"
    else:
        irx = yf.Ticker("^IRX").history(period="5d")  # T-bill 13 semanas, en %
        r = float(irx["Close"].iloc[-1]) / 100.0
        r_source = "^IRX (T-bill 13 sem.) / 100"

    # --- Vencimientos en ventana [~1 mes, ~1 anio] ---
    today = snap_ts.date()
    exp_all = tk.options
    chosen = []
    for e in exp_all:
        d = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if MIN_DAYS <= d <= MAX_DAYS:
            chosen.append((e, d))
    chosen.sort(key=lambda x: x[1])
    if len(chosen) > MAX_EXPIRIES:    # submuestreo uniforme manteniendo variedad
        idx = np.linspace(0, len(chosen) - 1, MAX_EXPIRIES).round().astype(int)
        chosen = [chosen[i] for i in idx]
    if len(chosen) < 5:
        print(f"AVISO: solo {len(chosen)} vencimientos en ventana; "
              f"el enunciado pide >= 5-6.")

    rows = []
    for exp, days in chosen:
        T = days / DAYS_PER_YEAR
        oc = tk.option_chain(exp)
        for typ, frame in (("call", oc.calls), ("put", oc.puts)):
            for _, o in frame.iterrows():
                rows.append({
                    "expiry": exp, "days": days, "T": T, "type": typ,
                    "K": float(o["strike"]),
                    "bid": float(o.get("bid", np.nan)),
                    "ask": float(o.get("ask", np.nan)),
                    "lastPrice": float(o.get("lastPrice", np.nan)),
                    "volume": float(o.get("volume", np.nan)),
                    "openInterest": float(o.get("openInterest", np.nan)),
                })

    df = pd.DataFrame(rows)
    meta = {
        "ticker": ticker,
        "snapshot_utc": snap_ts.isoformat(),
        "S0": S0,
        "r": r,
        "r_source": r_source,
        "n_expiries": len(chosen),
        "source": "yfinance / Yahoo Finance",
        "day_count": "ACT/365",
    }
    return df, meta


# ============================================================================
# 4. PIPELINE: limpiar, forward por T, IV/vega, congelar
# ============================================================================
def build_dataset(df_raw, meta, verbose=True):
    """Filtra, calcula forward/IV/vega por vencimiento y devuelve df limpio."""
    S0, r = meta["S0"], meta["r"]
    df = df_raw.copy()

    # --- mid y spread relativo ---
    df["bid"] = df["bid"].replace(0.0, np.nan)
    df["ask"] = df["ask"].replace(0.0, np.nan)
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]

    n0 = len(df)

    # --- (i) cotizaciones validas: bid/ask presentes y coherentes ---
    df = df[df["bid"].notna() & df["ask"].notna() & (df["ask"] >= df["bid"])]
    df = df[df["mid"] >= MIN_MID]

    # --- (ii) liquidez: interes abierto / volumen ---
    if REQUIRE_OI:
        oi = df["openInterest"].fillna(0.0)
        vol = df["volume"].fillna(0.0)
        df = df[(oi > 0) | (vol > 0)]

    # --- (iii) spread bid-ask no excesivo ---
    df = df[df["rel_spread"] <= MAX_REL_SPREAD]

    # --- (iv) cercania al dinero 0.8 <= K/S0 <= 1.2 ---
    df["moneyness"] = df["K"] / S0
    df = df[(df["moneyness"] >= MONEYNESS_LO) & (df["moneyness"] <= MONEYNESS_HI)]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Filtrado: {n0} -> {len(df)} cotizaciones.")

    if df.empty:
        raise RuntimeError("Tras el filtrado no quedan cotizaciones; relaja los filtros.")

    # --- forward implicito por vencimiento (paridad) y spot efectivo ---
    fwd_info = {}
    for T, g in df.groupby("T"):
        F_T, q_T, npar = implied_forward(g, r, T, S0)
        fwd_info[T] = (F_T, q_T, npar)

    df["F"]     = df["T"].map(lambda t: fwd_info[t][0])
    df["q_impl"] = df["T"].map(lambda t: fwd_info[t][1])
    df["S_adj"] = df["F"] * np.exp(-r * df["T"])   # spot efectivo (q absorbido)

    # --- IV de mercado y vega (con S_adj, r) ---
    iv, vega, otm = [], [], []
    for _, o in df.iterrows():
        ivi = implied_vol(o["mid"], o["S_adj"], o["K"], o["T"], r, o["type"])
        iv.append(ivi)
        vega.append(bs_vega(o["S_adj"], o["K"], o["T"], r, ivi) if np.isfinite(ivi) else np.nan)
        # OTM: put si K < F, call si K > F (lado mas liquido / menor sesgo americano)
        otm.append((o["type"] == "put" and o["K"] < o["F"]) or
                   (o["type"] == "call" and o["K"] >= o["F"]))
    df["iv_mkt"] = iv
    df["vega"]   = vega
    df["otm"]    = otm

    # descartar IVs que no invirtieron (fuera de cotas de no-arbitraje)
    n_iv = len(df)
    df = df[df["iv_mkt"].notna()].reset_index(drop=True)
    if verbose:
        print(f"IV valida: {n_iv} -> {len(df)} (se descartan las fuera de cotas).")

    return df, fwd_info


def report_counts(df, fwd_info, meta):
    """Tabla de cotizaciones por vencimiento tras el filtrado."""
    print("\n" + "=" * 78)
    print(f"SNAPSHOT {meta['ticker']}  |  {meta['snapshot_utc']}")
    print(f"S0 = {meta['S0']:.4f}   r = {meta['r']:.4%}  ({meta['r_source']})")
    print("=" * 78)
    print(f"{'expiry':<12} {'dias':>5} {'T':>7} {'calls':>6} {'puts':>6} "
          f"{'F_impl':>9} {'q_impl':>8} {'IV_atm':>8}")
    print("-" * 78)
    for T in sorted(df["T"].unique()):
        g = df[df["T"] == T]
        exp = g["expiry"].iloc[0]
        days = int(g["days"].iloc[0])
        nc = int((g["type"] == "call").sum())
        npu = int((g["type"] == "put").sum())
        F_T, q_T, _ = fwd_info[T]
        i_atm = (g["K"] - F_T).abs().idxmin()
        iv_atm = g.loc[i_atm, "iv_mkt"]
        print(f"{exp:<12} {days:>5} {T:>7.4f} {nc:>6} {npu:>6} "
              f"{F_T:>9.3f} {q_T:>8.4f} {iv_atm:>8.4f}")
    print("-" * 78)
    print(f"TOTAL cotizaciones: {len(df)}   "
          f"({(df['type']=='call').sum()} calls, {(df['type']=='put').sum()} puts)")
    print("=" * 78)


def save_snapshot(df_raw, df_clean, meta):
    cols = ["expiry", "days", "T", "type", "K", "bid", "ask", "mid",
            "volume", "openInterest", "rel_spread", "moneyness",
            "F", "q_impl", "S_adj", "iv_mkt", "vega", "otm"]
    df_raw.to_csv(OUT_RAW, index=False)
    df_clean[cols].to_csv(OUT_CLEAN, index=False)
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nArchivos escritos:\n  {OUT_RAW}\n  {OUT_CLEAN}\n  {OUT_META}")


# ============================================================================
# 5. MODO SELFTEST: cadena sintetica (sin red) para validar el pipeline
# ============================================================================
def synthetic_chain(meta):
    """Genera una cadena coherente con BS+dividendo q para probar el pipeline."""
    rng = np.random.default_rng(SEED)
    S0, r = meta["S0"], meta["r"]
    q_true = 0.035
    expiries = [(30, "2025-01-30"), (60, "2025-03-01"), (120, "2025-04-30"),
                (200, "2025-07-19"), (300, "2025-10-27"), (370, "2026-01-05")]
    rows = []
    for days, exp in expiries:
        T = days / DAYS_PER_YEAR
        F = S0 * np.exp((r - q_true) * T)
        S_adj = F * np.exp(-r * T)
        # sonrisa sintetica: skew + curvatura en log-moneyness
        for K in np.arange(round(0.7 * S0), round(1.3 * S0) + 1, 1.0):
            x = np.log(K / F)
            iv = 0.22 + 0.10 * x**2 - 0.05 * x      # smile/skew
            for typ in ("call", "put"):
                px = bs_price(S_adj, K, T, r, iv, typ)
                half = px * (0.01 + 0.02 * rng.random())   # spread sintetico
                rows.append({
                    "expiry": exp, "days": days, "T": T, "type": typ,
                    "K": float(K),
                    "bid": max(px - half, 0.0), "ask": px + half,
                    "lastPrice": px,
                    "volume": float(rng.integers(0, 500)),
                    "openInterest": float(rng.integers(0, 5000)),
                })
    meta["_q_true_synth"] = q_true
    return pd.DataFrame(rows)


def run_selftest():
    print(">>> MODO SELFTEST (cadena sintetica, sin red)\n")
    meta = {
        "ticker": TICKER, "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "S0": 90.0, "r": 0.0475, "r_source": "sintetico", "source": "selftest",
        "day_count": "ACT/365",
    }
    df_raw = synthetic_chain(meta)
    df_clean, fwd_info = build_dataset(df_raw, meta)
    report_counts(df_clean, fwd_info, meta)

    # --- verificaciones automaticas ---
    print("\nVERIFICACIONES:")
    # (a) forward implicito recupera q_true
    q_true = meta["_q_true_synth"]
    q_err = max(abs(fwd_info[T][1] - q_true) for T in fwd_info)
    print(f"  (a) q implicita vs q_true={q_true}: error max = {q_err:.2e} "
          f"{'OK' if q_err < 1e-6 else 'REVISAR'}")
    # (b) IV recuperada vs IV teorica usada al generar
    S0, r = meta["S0"], meta["r"]
    errs = []
    for _, o in df_clean.iterrows():
        x = np.log(o["K"] / o["F"])
        iv_th = 0.22 + 0.10 * x**2 - 0.05 * x
        errs.append(abs(o["iv_mkt"] - iv_th))
    print(f"  (b) IV recuperada vs IV teorica: error max = {max(errs):.2e} "
          f"{'OK' if max(errs) < 1e-3 else 'REVISAR'}")
    # (c) vega > 0 y finita
    print(f"  (c) vega finita y positiva: "
          f"{'OK' if (df_clean['vega'] > 0).all() else 'REVISAR'}")
    print("\nSelftest completado.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="corre el pipeline sobre datos sinteticos, sin red")

    # In Colab/Jupyter, sys.argv often contains kernel-specific arguments.
    # To avoid 'unrecognized arguments: -f' when running directly in a cell,
    # we explicitly pass an empty list of arguments to argparse.
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    print(f">>> Descargando cadena de {TICKER} desde Yahoo Finance...")
    df_raw, meta = fetch_chain_yfinance(TICKER)
    df_clean, fwd_info = build_dataset(df_raw, meta)
    report_counts(df_clean, fwd_info, meta)
    save_snapshot(df_raw, df_clean, meta)


if __name__ == "__main__":
    main()
