from __future__ import annotations
# src/pde_discretization/__init__.py
from .domain_mesh import SinaiBilliardMesh, build_sinai_mesh
from .shortley_weller import (
    assemble_shortley_weller_nonsymmetric,
    assemble_staircase_laplacian,
)
from .symmetrization import (
    SymmetricLaplacianSystem,
    assemble_symmetric_laplacian,
    recover_physical_eigenfunction,
)

__all__ = [
    "SinaiBilliardMesh",
    "build_sinai_mesh",
    "assemble_shortley_weller_nonsymmetric",
    "assemble_staircase_laplacian",
    "SymmetricLaplacianSystem",
    "assemble_symmetric_laplacian",
    "recover_physical_eigenfunction",
]