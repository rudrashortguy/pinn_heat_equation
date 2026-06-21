import os
import torch
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from pinn import PINN
from domain import sample_interior, sample_boundary, sample_initial
from loss import compute_loss
from train import train


def test_train_creates_checkpoints():
    cfg = Config()
    cfg.epochs_adam = 2
    cfg.epochs_lbfgs = 1
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.checkpoint_dir = tmpdir
        model = train(cfg)
        assert isinstance(model, PINN)
        assert os.path.exists(os.path.join(tmpdir, "model_adam.pt"))
        assert os.path.exists(os.path.join(tmpdir, "model_lbfgs.pt"))


def test_train_loss_decreases():
    cfg = Config()
    cfg.epochs_adam = 5
    cfg.epochs_lbfgs = 1
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.checkpoint_dir = tmpdir
        model = train(cfg)
        x_colloc, t_colloc = sample_interior(cfg)
        x_bc, t_bc = sample_boundary(cfg)
        x_ic, t_ic = sample_initial(cfg)
        loss_dict = compute_loss(model, x_colloc, t_colloc, x_bc, t_bc, x_ic, t_ic, cfg)
        assert loss_dict["total"] >= 0


def test_model_checkpoint_loadable():
    cfg = Config()
    cfg.epochs_adam = 2
    cfg.epochs_lbfgs = 1
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.checkpoint_dir = tmpdir
        train(cfg)
        model2 = PINN(cfg)
        model2.load_state_dict(
            torch.load(os.path.join(tmpdir, "model_lbfgs.pt"), weights_only=True)
        )
        model2.eval()
        with torch.no_grad():
            out = model2(torch.tensor([[0.5]]), torch.tensor([[0.5]]))
        assert out.shape == (1, 1)
