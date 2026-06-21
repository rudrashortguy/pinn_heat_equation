import torch
from pinn import PINN
from physics import pde_residual, analytical_solution
from config import Config


def compute_loss(
    model: PINN,
    x_colloc: torch.Tensor,
    t_colloc: torch.Tensor,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    cfg: Config,
) -> dict[str, torch.Tensor]:
    u_colloc = model(x_colloc, t_colloc)
    residual = pde_residual(u_colloc, x_colloc, t_colloc, cfg.alpha)
    loss_pde = torch.mean(residual**2)

    u_bc = model(x_bc, t_bc)
    loss_bc = torch.mean(u_bc**2)

    u_ic = model(x_ic, t_ic)
    u_exact_ic = analytical_solution(x_ic, t_ic, cfg.alpha)
    loss_ic = torch.mean((u_ic - u_exact_ic) ** 2)

    loss_total = loss_pde + cfg.lambda_bc * loss_bc + cfg.lambda_ic * loss_ic

    return {"pde": loss_pde, "bc": loss_bc, "ic": loss_ic, "total": loss_total}
