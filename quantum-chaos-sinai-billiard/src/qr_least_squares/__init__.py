from __future__ import annotations
# src/qr_least_squares/__init__.py
from .qr_factorization import (
    gram_schmidt_clasico,
    gram_schmidt_modificado,
    qr_householder,
    resolver_minimos_cuadrados_qr,
)

__all__ = [
    "gram_schmidt_clasico",
    "gram_schmidt_modificado",
    "qr_householder",
    "resolver_minimos_cuadrados_qr",
]