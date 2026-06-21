import torch
import numpy as np


def pde_residual(u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_t - alpha * u_xx
    return residual


def analytical_solution(x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.sin(np.pi * x) * torch.exp(-alpha * np.pi**2 * t)
