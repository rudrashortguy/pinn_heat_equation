import torch
import torch.nn as nn
from config import Config


class PINN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_features = 2
        for _ in range(cfg.n_hidden_layers):
            layers.append(nn.Linear(in_features, cfg.n_neurons))
            layers.append(nn.Tanh())
            in_features = cfg.n_neurons
        layers.append(nn.Linear(cfg.n_neurons, 1))
        self.layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=1)
        return self.layers(inp)
