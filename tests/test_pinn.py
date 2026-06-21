import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinn import PINN
from config import Config


def test_forward_pass_shape():
    cfg = Config()
    model = PINN(cfg)
    batch_size = 32
    x = torch.rand(batch_size, 1)
    t = torch.rand(batch_size, 1)
    u_pred = model(x, t)
    assert u_pred.shape == (batch_size, 1)


def test_forward_pass_gradients():
    cfg = Config()
    model = PINN(cfg)
    x = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)
    u_pred = model(x, t)
    grad_u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    grad_u_t = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    assert grad_u_x.shape == (10, 1)
    assert grad_u_t.shape == (10, 1)
    assert grad_u_x.requires_grad
    assert grad_u_t.requires_grad


def test_model_layers():
    cfg = Config()
    model = PINN(cfg)
    assert len(model.layers) == cfg.n_hidden_layers * 2 + 1
    for i in range(cfg.n_hidden_layers):
        assert model.layers[i * 2].out_features == cfg.n_neurons
        assert model.layers[i * 2].in_features == cfg.n_neurons if i > 0 else 2
