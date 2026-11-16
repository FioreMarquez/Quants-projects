from __future__ import annotations
import numpy as np
from scipy.stats import norm


def black_scholes_price(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put"
) -> float | np.ndarray:
    """Calcula el precio de una opcion europea usando la formula de Black-Scholes.

    Parameters
    ----------
    S : float u ndarray
        Precio actual del activo subyacente.
    K : float
        Precio de ejercicio (strike).
    T : float
        Tiempo a la madurez (en anos).
    r : float
        Tasa libre de riesgo anualizada.
    sigma : float
        Volatilidad del activo subyacente.
    option_type : str
        Tipo de opcion: 'put' o 'call'.

    Returns
    -------
    float u ndarray
        Valor analitico de la opcion europea.
    """
    S = np.asarray(S, dtype=float)

    # Manejo del caso limite t = T (madurez)
    if T <= 0:
        if option_type.lower() == "put":
            return np.maximum(K - S, 0.0)
        elif option_type.lower() == "call":
            return np.maximum(S - K, 0.0)
        else:
            raise ValueError("option_type debe ser 'put' o 'call'.")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    elif option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        raise ValueError("option_type debe ser 'put' o 'call'.")

    return price


def black_scholes_greeks(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put"
) -> dict[str, float | np.ndarray]:
    """Calcula las Griegas principales (Delta y Gamma) para una opcion europea.

    Returns
    -------
    dict
        Diccionario con las claves 'delta' y 'gamma'.
    """
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Gamma es identica para calls y puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    if option_type.lower() == "put":
        delta = norm.cdf(d1) - 1.0
    elif option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        raise ValueError("option_type debe ser 'put' o 'call'.")

    return {"delta": delta, "gamma": gamma}


if __name__ == "__main__":
    # Parametros del Caso de Prueba (Forsyth & Vetzal 2002, Tabla 1)
    S_test = 100.0
    K_test = 100.0
    T_test = 0.25
    r_test = 0.10
    sigma_test = 0.80

    price_put = black_scholes_price(S_test, K_test, T_test, r_test, sigma_test, option_type="put")
    greeks_put = black_scholes_greeks(S_test, K_test, T_test, r_test, sigma_test, option_type="put")

    print(f"--- Validacion de Formula Analitica Black-Scholes ---")
    print(f"Parametros: S={S_test}, K={K_test}, T={T_test}, r={r_test}, sigma={sigma_test}")
    print(f"Valor del Put Europeo Analitico: {price_put:.5f}")  # Esperado: ~14.45191
    print(f"Delta: {greeks_put['delta']:.5f}")
    print(f"Gamma: {greeks_put['gamma']:.5f}")