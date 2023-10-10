import pytest
import torch
from sympy import poly, symbols

from src.simulated_bifurcation.polynomial.polynomial_compiler import (
    Order2MultivariatePolynomialCompiler as O2MPC,
)


def test_polynomial_compiler():
    x, y, z = symbols("x y z")
    polynomial = poly(x * y - 2 * y * z + 3 * x**2 - z + 2)
    constant, vector, matrix = O2MPC(polynomial).compile()
    assert torch.equal(torch.tensor([2.0]), constant)
    assert torch.equal(torch.tensor([0.0, 0.0, -1.0]), vector)
    assert torch.equal(
        torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]), matrix
    )


def test_wrong_degree():
    x, y = symbols("x y")
    with pytest.raises(ValueError, match="Expected degree 2 polynomial, got 1."):
        O2MPC(poly(x + 4 * y))
    with pytest.raises(ValueError, match="Expected degree 2 polynomial, got 3."):
        O2MPC(poly(x**2 * y - 5 * x * y + 2 * y + 3))
