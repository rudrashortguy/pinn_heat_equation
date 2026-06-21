import os
import torch
import torch.optim as optim
from config import Config
from pinn import PINN
from domain import sample_interior, sample_boundary, sample_initial
from loss import compute_loss


def train(cfg: Config | None = None):
    if cfg is None:
        cfg = Config()
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model = PINN(cfg)
    optimizer_adam = optim.Adam(model.parameters(), lr=cfg.lr_adam)

    x_colloc, t_colloc = sample_interior(cfg)
    x_bc, t_bc = sample_boundary(cfg)
    x_ic, t_ic = sample_initial(cfg)

    print("Starting Adam optimization...")
    model.train()
    for epoch in range(cfg.epochs_adam):
        optimizer_adam.zero_grad()
        loss_dict = compute_loss(model, x_colloc, t_colloc, x_bc, t_bc, x_ic, t_ic, cfg)
        loss_dict["total"].backward()
        optimizer_adam.step()

        if (epoch + 1) % 1000 == 0:
            print(
                f"Adam Epoch {epoch+1:5d}/{cfg.epochs_adam} | "
                f"PDE: {loss_dict['pde'].item():.6e} | "
                f"BC: {loss_dict['bc'].item():.6e} | "
                f"IC: {loss_dict['ic'].item():.6e} | "
                f"Total: {loss_dict['total'].item():.6e}"
            )

    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "model_adam.pt"))
    print("Adam finished. Starting L-BFGS fine-tuning...")

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=cfg.lr_lbfgs,
        max_iter=cfg.epochs_lbfgs,
        tolerance_change=1.0e-12,
        history_size=50,
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss_dict = compute_loss(model, x_colloc, t_colloc, x_bc, t_bc, x_ic, t_ic, cfg)
        loss_dict["total"].backward()
        return loss_dict["total"]

    optimizer_lbfgs.step(closure)
    final_loss = closure().item()

    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "model_lbfgs.pt"))
    print(f"L-BFGS finished. Final total loss: {final_loss:.6e}")
    return model


if __name__ == "__main__":
    train()
