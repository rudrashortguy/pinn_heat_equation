import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import Config
from pinn import PINN
from physics import analytical_solution


def evaluate():
    cfg = Config()
    device = torch.device("cpu")
    model = PINN(cfg)
    model.load_state_dict(
        torch.load("checkpoints/model_lbfgs.pt", map_location=device, weights_only=True)
    )
    model.eval()

    nx, nt = 100, 100
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(X_flat, T_flat).reshape(nx, nt)

    u_exact = analytical_solution(X_flat, T_flat, cfg.alpha).reshape(nx, nt)

    l2_error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    print(f"L2 relative error: {l2_error.item():.6e}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time_indices = [0, nt // 4, nt // 2, 3 * nt // 4]
    time_labels = [0.0, 0.25, 0.5, 0.75]

    for idx, (ti, tl) in enumerate(zip(time_indices, time_labels)):
        ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]
        ax.plot(x.numpy(), u_pred[:, ti].numpy(), "b-", label="PINN", linewidth=2)
        ax.plot(x.numpy(), u_exact[:, ti].numpy(), "r--", label="Analytical", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"t = {tl}")
        ax.legend()
        ax.grid(True)

    for idx in range(4, 6):
        ax = axes[1, idx - 3] if idx < 3 else axes[1, idx - 3]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("output/comparison.png", dpi=150)
    print("Comparison plot saved to output/comparison.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.contourf(X.numpy(), T.numpy(), u_pred.numpy(), levels=50, cmap="viridis")
    ax1.set_title("PINN Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.contourf(X.numpy(), T.numpy(), u_exact.numpy(), levels=50, cmap="viridis")
    ax2.set_title("Analytical Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig("output/heatmaps.png", dpi=150)
    print("Heatmaps saved to output/heatmaps.png")

    return l2_error.item()


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    evaluate()
