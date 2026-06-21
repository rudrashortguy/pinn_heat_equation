import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics import pde_residual, analytical_solution


def test_pde_residual_zero_for_analytical():
    alpha = 0.1
    x = torch.linspace(0, 1, 50, requires_grad=True)
    t = torch.linspace(0, 1, 50, requires_grad=True)
    X, T = torch.meshgrid(x, t, indexing="ij")
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)

    u = analytical_solution(X, T, alpha)
    residual = pde_residual(u, X, T, alpha)

    assert residual.shape == (2500, 1)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_analytical_solution_shape():
    alpha = 0.1
    x = torch.tensor([[0.5]])
    t = torch.tensor([[0.0]])
    u = analytical_solution(x, t, alpha)
    assert u.shape == (1, 1)


def test_analytical_solution_boundary():
    alpha = 0.1
    x = torch.tensor([[0.0], [1.0]])
    t = torch.tensor([[0.5]])
    u = analytical_solution(x, t, alpha)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)


def test_analytical_solution_initial():
    alpha = 0.1
    x = torch.linspace(0, 1, 10).reshape(-1, 1)
    t = torch.zeros(10, 1)
    u = analytical_solution(x, t, alpha)
    expected = torch.sin(np.pi * x)
    assert torch.allclose(u, expected, atol=1e-5)
