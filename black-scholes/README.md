# Valuación de Opciones Americanas mediante Elementos Finitos y Penalización

Implementación numérica en Python para la valuación de opciones *put* americanas bajo el modelo de Black-Scholes mediante el método de elementos finitos lineales a trozos ($P_1$) y una formulación de penalización resuelta con iteración de Newton generalizada, siguiendo el enfoque de Forsyth & Vetzal (2002).

## Metodología Numérica

- **Discretización espacial ($P_1$):** malla unidimensional sobre $[0, S_{\max}]$ con un nodo ubicado exactamente en el precio de ejercicio $S = K$, condensación de masa (*mass lumping*) y estabilización aguas arriba local para garantizar la propiedad de $M$-matriz y evitar oscilaciones espurias.
- **Método de penalización:** transformación del problema de complementariedad lineal (LCP) en una ecuación diferencial parcial no lineal mediante un término de penalización activo cuando $V < V^*$, resuelta en cada paso mediante iteración de Newton no suave.
- **Integración temporal:** esquema de Crank-Nicolson ($\theta = 1/2$) precedido por dos pasos iniciales implícitos puros ($\theta = 0$, suavizado de Rannacher) para amortiguar la falta de regularidad del pago en $S = K$, junto con un selector adaptativo de paso temporal $\Delta\tau$.
- **Validación cruzada:** contraste frente a la fórmula analítica europea de Black-Scholes y un árbol binomial de Cox-Ross-Rubinstein (CRR) de alta resolución.

## Estructura del Proyecto

```text
black-scholes/
├── main.py                # Script principal de experimentos y tablas de convergencia
├── requirements.txt       # Dependencias del entorno
├── README.md              # Documentación del proyecto
└── src/
    ├── fem.py             # Generación de malla, ensamblaje FEM P1, M-matriz y griegas
    ├── penalty.py         # Solver de Newton generalizado para el paso penalizado
    ├── timestepping.py    # Integración temporal (Rannacher + Crank-Nicolson + paso adaptativo)
    ├── binomial.py        # Árbol binomial CRR para referencia americana
    └── black_scholes.py   # Valuación y griegas analíticas europeas
