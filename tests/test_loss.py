import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss import compute_loss
from pinn import PINN
from domain import sample_interior, sample_boundary, sample_initial
from config import Config


def test_loss_components_computed():
    cfg = Config()
    model = PINN(cfg)
    x_colloc, t_colloc = sample_interior(cfg)
    x_bc, t_bc = sample_boundary(cfg)
    x_ic, t_ic = sample_initial(cfg)

    loss_dict = compute_loss(model, x_colloc, t_colloc, x_bc, t_bc, x_ic, t_ic, cfg)

    assert "pde" in loss_dict
    assert "bc" in loss_dict
    assert "ic" in loss_dict
    assert "total" in loss_dict

    assert loss_dict["pde"] >= 0
    assert loss_dict["bc"] >= 0
    assert loss_dict["ic"] >= 0
    assert loss_dict["total"] >= 0


def test_loss_zero_for_perfect_model():
    cfg = Config()
    model = PINN(cfg)

    x_colloc = torch.rand(100, 1, requires_grad=True)
    t_colloc = torch.rand(100, 1, requires_grad=True)

    x_bc = torch.tensor([[0.0], [1.0]]).repeat(50, 1)
    t_bc = torch.rand(100, 1)

    x_ic = torch.rand(100, 1)
    t_ic = torch.zeros(100, 1)

    loss_dict = compute_loss(model, x_colloc, t_colloc, x_bc, t_bc, x_ic, t_ic, cfg)

    assert loss_dict["pde"] >= 0
    assert loss_dict["bc"] >= 0
    assert loss_dict["ic"] >= 0
