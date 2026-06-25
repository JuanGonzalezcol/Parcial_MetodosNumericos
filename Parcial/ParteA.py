# -*- coding: utf-8 -*-
r"""
Parcial - Calibracion del modelo Markov-modulado
================================================
PARTE A - Formula analitica por mezcla (Teorema 1)

Precio de la call/put europea condicionado al regimen inicial i, como una
MEZCLA de Black-Scholes sobre la ley de la varianza integrada A_{0,T}:

    C_i(0,s) = e^{-lam_i T} BS(s,K,T,r(a*_i); a*_i)
               + \int_{a-}^{a+} BS(s,K,T,r(a); a) g_i^A(a,T) da,

con a- = sigma0^2 T, a+ = sigma1^2 T, a*_0=a-, a*_1=a+, y tasa integrada del
camino R(a) = r0 T0 + r1 T1, T1=(a - sigma0^2 T)/Dsig, T0=T-T1, r(a)=R(a)/T.

Vale para r0 != r1 (la tasa integrada R(a) depende de a). Con r0=r1=r se
recupera R(a)=rT (BS de tasa constante).

Densidad continua de A_{0,T} (parte tipo Bessel), con Dsig = sigma1^2-sigma0^2,
u = a - sigma0^2 T, v = sigma1^2 T - a:
    eta(a)   = (2/Dsig) sqrt(lam0 lam1 u v)
    Gamma(a) = (lam0 v + lam1 u)/Dsig
    g_0^A = e^{-Gamma}/Dsig [ lam0 I0(eta) + sqrt(lam0 lam1 v/u) I1(eta) ]
    g_1^A = e^{-Gamma}/Dsig [ lam1 I0(eta) + sqrt(lam0 lam1 u/v) I1(eta) ]
Atomos: Q_0(A=a-) = e^{-lam0 T},  Q_1(A=a+) = e^{-lam1 T}.

Notas numericas:
  * Aunque g_i^A es ACOTADA en los bordes (sqrt(v/u) I1(eta) -> limite finito
    porque I1(eta)~eta/2 ~ sqrt(u v)), el integrando tiene un cusp tipo sqrt en
    a±. Usamos el cambio de variable a = c + h sin(phi) (c=(a-+a+)/2,
    h=(a+-a-)/2): da = h cos(phi) dphi anula el integrando en los bordes y lo
    vuelve suave -> quad converge rapido y sin tocar los extremos (u,v>0).
  * Usamos i0e/i1e (Bessel escaladas, i0e(x)=e^{-x}I0(x)) y factorizamos
    exp(eta - Gamma), que esta acotado por 0 (AM-GM), evitando overflow.
  * Caso singular sigma0 -> sigma1 (Dsig -> 0): el soporte colapsa y la ley
    degenera a una masa en sigma^2 T; devolvemos BS plana.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e, i1e
from scipy.stats import norm


# ----------------------------------------------------------------------------
# Black-Scholes parametrizado por tasa INTEGRADA R (=r*T) y varianza TOTAL a (=sigma^2 T)
# ----------------------------------------------------------------------------
def bs_integrated(s, K, T, R, a, option_type="call"):
    """
    BS con tasa integrada R y varianza total a:
        d1,2 = (ln(s/K) + R +- a/2)/sqrt(a),  descuento e^{-R}.
    """
    s, K = float(s), float(K)
    if a <= 0:                          # varianza nula: pago deterministico
        fwd = s * np.exp(R)             # forward bajo tasa integrada R
        disc = np.exp(-R)
        if option_type == "call":
            return disc * max(fwd - K, 0.0)
        else:
            return disc * max(K - fwd, 0.0)
    sqrt_a = np.sqrt(a)
    d1 = (np.log(s / K) + R + 0.5 * a) / sqrt_a
    d2 = d1 - sqrt_a
    disc = np.exp(-R)
    if option_type == "call":
        return s * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        return K * disc * norm.cdf(-d2) - s * norm.cdf(-d1)


# ----------------------------------------------------------------------------
# Densidad continua de la varianza integrada A_{0,T} (regimen inicial i)
# ----------------------------------------------------------------------------
def occupation_density(a, T, sigma0, sigma1, lam0, lam1, regime=0):
    """g_i^A(a,T) en el interior (a-, a+).  Asume sigma0 < sigma1."""
    dsig = sigma1**2 - sigma0**2
    a_minus = sigma0**2 * T
    a_plus = sigma1**2 * T

    u = a - a_minus            # = a - sigma0^2 T  >= 0
    v = a_plus - a             # = sigma1^2 T - a  >= 0
    eps = 1e-300
    u = max(u, eps)
    v = max(v, eps)

    eta = (2.0 / dsig) * np.sqrt(lam0 * lam1 * u * v)
    Gamma = (lam0 * v + lam1 * u) / dsig

    # factor comun exp(eta - Gamma), acotado por 0  ->  estable
    pref = np.exp(eta - Gamma) / dsig

    if regime == 0:
        term = lam0 * i0e(eta) + np.sqrt(lam0 * lam1 * v / u) * i1e(eta)
    else:
        term = lam1 * i0e(eta) + np.sqrt(lam0 * lam1 * u / v) * i1e(eta)
    return pref * term


# ----------------------------------------------------------------------------
# Precio analitico por mezcla (Teorema 1)
# ----------------------------------------------------------------------------
def price_analytic_markov(S0, K, T, r0, r1, sigma0, sigma1, lam0, lam1,
                          regime=0, option_type="call"):
    """
    Precio europeo por la formula de mezcla (Teorema 1).
    Vale para r0 != r1: usa la tasa integrada del camino R(a)=r0 T0 + r1 T1.
    Asume sigma0 < sigma1 (si no, reetiqueta automaticamente).
    Devuelve: precio (float).
    """
    # romper simetria de etiqueta: garantizar sigma0 < sigma1
    if sigma0 > sigma1:
        sigma0, sigma1 = sigma1, sigma0
        lam0, lam1 = lam1, lam0
        r0, r1 = r1, r0
        regime = 1 - regime

    dsig = sigma1**2 - sigma0**2
    a_minus = sigma0**2 * T
    a_plus = sigma1**2 * T

    # --- caso singular: sigma0 -> sigma1 (soporte colapsa) -> BS plana ---
    if dsig <= 1e-12 * max(1.0, sigma1**2 * T) or T <= 0:
        R = r0 * T
        return bs_integrated(S0, K, T, R, sigma0**2 * T, option_type)

    # --- atomo del borde (evento "sin cambio de regimen") ---
    if regime == 0:
        lam_i, a_star, R_star = lam0, a_minus, r0 * T
    else:
        lam_i, a_star, R_star = lam1, a_plus, r1 * T
    atom = np.exp(-lam_i * T) * bs_integrated(S0, K, T, R_star, a_star, option_type)

    # --- integral continua con cambio de variable seno ---
    c = 0.5 * (a_minus + a_plus)
    h = 0.5 * (a_plus - a_minus)

    def integrand(phi):
        a = c + h * np.sin(phi)
        T1 = (a - a_minus) / dsig          # tiempo de ocupacion del regimen 1
        R = r0 * (T - T1) + r1 * T1        # tasa integrada del camino
        bs = bs_integrated(S0, K, T, R, a, option_type)
        g = occupation_density(a, T, sigma0, sigma1, lam0, lam1, regime)
        return bs * g * h * np.cos(phi)

    integral, _ = quad(integrand, -np.pi / 2, np.pi / 2, limit=200)
    return atom + integral


# ----------------------------------------------------------------------------
# Masa total de la ley de A_{0,T} (atomo + integral de la densidad) -> debe = 1
# ----------------------------------------------------------------------------
def total_mass(T, sigma0, sigma1, lam0, lam1, regime=0):
    if sigma0 > sigma1:
        sigma0, sigma1 = sigma1, sigma0
        lam0, lam1 = lam1, lam0
        regime = 1 - regime
    dsig = sigma1**2 - sigma0**2
    a_minus, a_plus = sigma0**2 * T, sigma1**2 * T
    lam_i = lam0 if regime == 0 else lam1
    atom = np.exp(-lam_i * T)

    c = 0.5 * (a_minus + a_plus)
    h = 0.5 * (a_plus - a_minus)

    def integrand(phi):
        a = c + h * np.sin(phi)
        g = occupation_density(a, T, sigma0, sigma1, lam0, lam1, regime)
        return g * h * np.cos(phi)

    integral, _ = quad(integrand, -np.pi / 2, np.pi / 2, limit=200)
    return atom + integral


# ============================================================================
# PRUEBAS INTERNAS (enunciado, Parte A)
# ============================================================================
if __name__ == "__main__":
    # parametros de prueba del Taller 2
    S0 = K = 100.0
    T = 0.5
    r = 0.03
    sigma0, sigma1 = 0.15, 0.40
    lam0, lam1 = 2.0, 5.0

    print("=" * 70)
    print("PARTE A - Pruebas internas (params Taller 2)")
    print(f"S0=K={S0}, T={T}, r={r}, sigma=({sigma0},{sigma1}), lam=({lam0},{lam1})")
    print("=" * 70)

    # ----- (a) Masa total = 1 en ambos regimenes -----
    print("\n(a) Masa total  e^{-lam_i T} + int g_i^A da = 1")
    for reg in (0, 1):
        m = total_mass(T, sigma0, sigma1, lam0, lam1, regime=reg)
        atom = np.exp(-(lam0 if reg == 0 else lam1) * T)
        print(f"    regimen {reg}: masa = {m:.10f}   "
              f"(atomo={atom:.4f}, cont={m-atom:.4f})   "
              f"|error| = {abs(m-1.0):.2e}  "
              f"{'OK' if abs(m-1.0) < 1e-6 else 'REVISAR'}")

    # ----- (c) Cota (r0=r1): BS(a-) <= C_i <= BS(a+) -----
    print("\n(c) Cota sandwich  BS(.;a-) <= C_i <= BS(.;a+)  (r0=r1)")
    bs_lo = bs_integrated(S0, K, T, r * T, sigma0**2 * T, "call")   # BS con sigma0
    bs_hi = bs_integrated(S0, K, T, r * T, sigma1**2 * T, "call")   # BS con sigma1
    for reg in (0, 1):
        Ci = price_analytic_markov(S0, K, T, r, r, sigma0, sigma1,
                                   lam0, lam1, regime=reg, option_type="call")
        ok = bs_lo - 1e-9 <= Ci <= bs_hi + 1e-9
        print(f"    regimen {reg}: BS(a-)={bs_lo:.6f} <= C_i={Ci:.6f} "
              f"<= BS(a+)={bs_hi:.6f}   {'OK' if ok else 'FALLA'}")

    # ----- (b) Caso singular sigma0 -> sigma1: precio -> BS plana -----
    print("\n(b) Caso singular sigma0 -> sigma1: C_i -> BS plana en sigma")
    sig = 0.25
    bs_flat = bs_integrated(S0, K, T, r * T, sig**2 * T, "call")
    print(f"    BS plana (sigma={sig}): {bs_flat:.6f}")
    for d in (1e-2, 1e-4, 1e-6, 1e-10):
        s0_, s1_ = sig, sig + d
        Ci = price_analytic_markov(S0, K, T, r, r, s0_, s1_, lam0, lam1,
                                   regime=0, option_type="call")
        print(f"    sigma1-sigma0={d:.0e}: C_0={Ci:.6f}  "
              f"|dif BS|={abs(Ci-bs_flat):.2e}")

    # ----- Verificacion extra: put por mezcla vs paridad por regimen -----
    print("\n(extra) Paridad call-put por regimen (r0=r1):  C_i - P_i = s - K e^{-rT}")
    for reg in (0, 1):
        Ci = price_analytic_markov(S0, K, T, r, r, sigma0, sigma1,
                                   lam0, lam1, regime=reg, option_type="call")
        Pi = price_analytic_markov(S0, K, T, r, r, sigma0, sigma1,
                                   lam0, lam1, regime=reg, option_type="put")
        lhs = Ci - Pi
        rhs = S0 - K * np.exp(-r * T)
        print(f"    regimen {reg}: C-P={lhs:.8f}  s-Ke^-rT={rhs:.8f}  "
              f"|error|={abs(lhs-rhs):.2e}")

    # ----- Sanidad: sin regimen efectivo (lam=0, regimen 0) = BS(sigma0) -----
    print("\n(sanidad) lam0=lam1=0, regimen 0  ->  BS(sigma0) puro")
    Ci = price_analytic_markov(S0, K, T, r, r, sigma0, sigma1, 0.0, 0.0,
                               regime=0, option_type="call")
    print(f"    C_0 = {Ci:.6f}   BS(sigma0) = {bs_lo:.6f}   "
          f"|error| = {abs(Ci-bs_lo):.2e}")