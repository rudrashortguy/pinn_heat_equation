import torch
from config import Config


def sample_interior(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(cfg.n_collocation, 1)
    t = torch.rand(cfg.n_collocation, 1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_boundary(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    half = cfg.n_boundary // 2
    x = torch.cat([torch.zeros(half, 1), torch.ones(cfg.n_boundary - half, 1)])
    t = torch.rand(cfg.n_boundary, 1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_initial(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(cfg.n_initial, 1)
    t = torch.zeros(cfg.n_initial, 1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t
