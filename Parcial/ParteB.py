# -*- coding: utf-8 -*-
r"""
Parcial - Calibracion del modelo Markov-modulado
================================================
PARTE B - Validacion cruzada de las tres rutas

Antes de calibrar nada, las tres implementaciones deben calcular lo mismo:
  Ruta A: price_analytic_markov  (Parte A, formula de mezcla, Teorema 1)
  Ruta B: cos_method             (Taller 2, metodo COS)            <- motor de calib.
  Ruta C: fd_european_put_markov (Taller 3, EDP theta-esquema)

Reconciliamos todo a CALL usando la paridad por regimen (r0=r1=r):
    C_i(0,s) - P_i(0,s) = s - K e^{-rT},  i in {0,1}.
La Ruta C entrega put -> convertimos a call:  C_i = P_i + s - K e^{-rT}.

Para K in {90,100,110} y ambos regimenes tabulamos A, B, C y reportamos
|A-B|, |A-C|, |B-C|.  Deben coincidir dentro de ~1e-3 (la tolerancia de C la
limita la malla).
"""

import numpy as np
from scipy.stats import norm
from    ParteA import price_analytic_markov

# ============================================================================
# Codigo del Taller 2  (Ruta B: COS)  -- reutilizado
# ============================================================================
def char_func_markov(z, T, r0, r1, sigma0, sigma1, lam0, lam1, regime=0):
    """Funcion caracteristica del log-precio bajo Q (version estable, Taller 2)."""
    z = np.asarray(z, dtype=np.complex128)
    mu_hat0 = r0 - 0.5 * sigma0**2
    mu_hat1 = r1 - 0.5 * sigma1**2
    mu_plus = (mu_hat0 + mu_hat1) / 2
    mu_minus = (mu_hat0 - mu_hat1) / 2
    sigma_plus = (sigma0**2 + sigma1**2) / 4
    sigma_minus = (sigma0**2 - sigma1**2) / 4
    lam_plus = (lam0 + lam1) / 2
    lam_minus = (lam0 - lam1) / 2

    rho_plus = 1j * z * mu_plus - sigma_plus * z**2 - lam_plus
    rho_minus = 1j * z * mu_minus - sigma_minus * z**2 - lam_minus

    D = rho_minus**2 + lam0 * lam1
    sqrt_D = np.sqrt(D)
    safe_sqrt_D = np.where(sqrt_D == 0, 1.0, sqrt_D)

    if regime == 0:
        c = (rho_minus + lam0) / safe_sqrt_D
        limit_val = np.exp(T * rho_plus) * (1.0 + (rho_minus + lam0) * T)
    else:
        c = -(rho_minus - lam1) / safe_sqrt_D
        limit_val = np.exp(T * rho_plus) * (1.0 - (rho_minus - lam1) * T)

    term1 = 0.5 * np.exp(T * (rho_plus + sqrt_D)) * (1.0 + c)
    term2 = 0.5 * np.exp(T * (rho_plus - sqrt_D)) * (1.0 - c)
    stable_phi = term1 + term2
    return np.where(sqrt_D == 0, limit_val, stable_phi)


def cos_method(S0, K, T, r, sigma0, sigma1, lam0, lam1,
               regime=0, N=256, L=10, option_type="call"):
    """Valoracion europea por COS en el modelo Markov-modulado (Taller 2)."""
    K = np.atleast_1d(np.float64(K))
    x = np.log(S0 / K)

    pi0 = lam1 / (lam0 + lam1) if (lam0 + lam1) > 0 else 0.5
    mu_avg = pi0 * (r - 0.5 * sigma0**2) + (1 - pi0) * (r - 0.5 * sigma1**2)
    var_avg = pi0 * sigma0**2 + (1 - pi0) * sigma1**2

    c1 = mu_avg * T
    c2 = var_avg * T
    c4 = 0.0
    a = c1 - L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))
    b = c1 + L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))

    k_arr = np.arange(N)
    z_vals = k_arr * np.pi / (b - a)
    phi_vals = char_func_markov(z_vals, T, r, r, sigma0, sigma1, lam0, lam1, regime)

    def chi_k(c, d, k_arr):
        arg_d = k_arr * np.pi * (d - a) / (b - a)
        arg_c = k_arr * np.pi * (c - a) / (b - a)
        denom = 1 + (k_arr * np.pi / (b - a))**2
        num = (np.cos(arg_d) * np.exp(d) - np.cos(arg_c) * np.exp(c)
               + k_arr * np.pi / (b - a)
               * (np.sin(arg_d) * np.exp(d) - np.sin(arg_c) * np.exp(c)))
        return num / denom

    def psi_k(c, d, k_arr):
        result = np.zeros_like(k_arr, dtype=float)
        arg_d = k_arr * np.pi * (d - a) / (b - a)
        arg_c = k_arr * np.pi * (c - a) / (b - a)
        nz = k_arr != 0
        result[nz] = ((np.sin(arg_d[nz]) - np.sin(arg_c[nz]))
                      * (b - a) / (k_arr[nz] * np.pi))
        result[~nz] = d - c
        return result

    if option_type == "call":
        Vk = 2 / (b - a) * (chi_k(0, b, k_arr) - psi_k(0, b, k_arr))
    else:
        Vk = 2 / (b - a) * (-chi_k(a, 0, k_arr) + psi_k(a, 0, k_arr))

    prices = np.zeros(len(K))
    for j, x_val in enumerate(x):
        exp_term = np.exp(1j * z_vals * x_val - 1j * k_arr * np.pi * a / (b - a))
        cos_coeffs = np.real(phi_vals * exp_term)
        cos_coeffs[0] *= 0.5
        prices[j] = K[j] * np.exp(-r * T) * np.sum(cos_coeffs * Vk)
    return prices


# ============================================================================
# Codigo del Taller 3  (Ruta C: EDP theta-esquema)  -- reutilizado
# ============================================================================
from scipy.sparse import diags, identity, bmat
from scipy.sparse.linalg import splu


def fd_european_put_markov(S0, K, T, r0, r1, sigma0, sigma1, lam0, lam1,
                           regime=0, Smax=None, M=200, Nt=200, theta=0.5):
    """Diferencias finitas (theta-esquema) para put europea Markov-modulada (Taller 3)."""
    if Smax is None:
        Smax = 4.0 * K
    dS = Smax / M
    dtau = T / Nt
    S = np.linspace(0.0, Smax, M + 1)
    j = np.arange(M + 1)

    a0 = 0.5 * sigma0**2 * j**2 - 0.5 * r0 * j
    b0 = -sigma0**2 * j**2 - r0 - lam0
    c0 = 0.5 * sigma0**2 * j**2 + 0.5 * r0 * j
    a1 = 0.5 * sigma1**2 * j**2 - 0.5 * r1 * j
    b1 = -sigma1**2 * j**2 - r1 - lam1
    c1 = 0.5 * sigma1**2 * j**2 + 0.5 * r1 * j

    L0 = diags([a0[1:M], b0[:M], c0[:M-1]], [-1, 0, 1], shape=(M, M), format="csr")
    L1 = diags([a1[1:M], b1[:M], c1[:M-1]], [-1, 0, 1], shape=(M, M), format="csr")
    Iblock = identity(M, format="csr")
    L = bmat([[L0, lam0 * Iblock], [lam1 * Iblock, L1]], format="csr")

    payoff = np.maximum(K - S[:M], 0.0)
    U = np.concatenate([payoff, payoff])

    I2M = identity(2 * M, format="csr")
    A_imp = (I2M - theta * dtau * L).tocsc()
    B_exp = (I2M + (1.0 - theta) * dtau * L).tocsr()
    lu = splu(A_imp)
    for _ in range(Nt):
        U = lu.solve(B_exp @ U)

    V_grid = U[:M] if regime == 0 else U[M:]
    return float(np.interp(S0, S[:M], V_grid))


# ============================================================================
# Referencia BS (para contexto)
# ============================================================================
def bs_call(S0, K, T, r, sigma):
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sqrtT
    d2 = d1 - sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================================
# TABLA DE VALIDACION
# ============================================================================
if __name__ == "__main__":
    # parametros de prueba del Taller 2
    S0 = 100.0
    T = 0.5
    r = 0.03
    sigma0, sigma1 = 0.15, 0.40
    lam0, lam1 = 2.0, 5.0
    strikes = [90.0, 100.0, 110.0]

    # malla fina para la EDP (tolerancia de C limitada por la malla)
    M_fd, Nt_fd = 800, 800

    print("=" * 92)
    print("PARTE B - Validacion cruzada de las tres rutas (call europea)")
    print(f"S0={S0}, T={T}, r={r}, sigma=({sigma0},{sigma1}), lam=({lam0},{lam1}); "
          f"EDP: M={M_fd}, Nt={Nt_fd}, theta=0.5")
    print("=" * 92)
    header = (f"{'regimen':>7} {'K':>6} | {'A (analitica)':>14} {'B (COS)':>12} "
              f"{'C (EDP)':>12} | {'|A-B|':>9} {'|A-C|':>9} {'|B-C|':>9}")
    print(header)
    print("-" * 92)

    disc = lambda T: np.exp(-r * T)
    max_dev = 0.0
    for reg in (0, 1):
        for K in strikes:
            # Ruta A: call directa
            A = price_analytic_markov(S0, K, T, r, r, sigma0, sigma1,
                                      lam0, lam1, regime=reg, option_type="call")
            # Ruta B: COS, call directa
            B = float(cos_method(S0, K, T, r, sigma0, sigma1, lam0, lam1,
                                 regime=reg, option_type="call")[0])
            # Ruta C: EDP da put -> call por paridad por regimen
            P_fd = fd_european_put_markov(S0, K, T, r, r, sigma0, sigma1,
                                          lam0, lam1, regime=reg,
                                          M=M_fd, Nt=Nt_fd, theta=0.5)
            C = P_fd + S0 - K * np.exp(-r * T)

            dAB, dAC, dBC = abs(A - B), abs(A - C), abs(B - C)
            max_dev = max(max_dev, dAB, dAC, dBC)
            print(f"{reg:>7} {K:>6.0f} | {A:>14.6f} {B:>12.6f} {C:>12.6f} | "
                  f"{dAB:>9.2e} {dAC:>9.2e} {dBC:>9.2e}")
        print("-" * 92)

    print(f"Desviacion maxima entre rutas: {max_dev:.2e}   "
          f"{'OK (<1e-3)' if max_dev < 1e-3 else 'REVISAR malla EDP / N de COS'}")
    print("=" * 92)