import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain import sample_interior, sample_boundary, sample_initial
from config import Config


def test_sample_interior_size():
    cfg = Config()
    x, t = sample_interior(cfg)
    assert x.shape == (cfg.n_collocation, 1)
    assert t.shape == (cfg.n_collocation, 1)


def test_sample_interior_range():
    cfg = Config()
    x, t = sample_interior(cfg)
    assert (x >= 0).all() and (x <= 1).all()
    assert (t >= 0).all() and (t <= 1).all()


def test_sample_boundary_size():
    cfg = Config()
    x, t = sample_boundary(cfg)
    assert x.shape == (cfg.n_boundary, 1)
    assert t.shape == (cfg.n_boundary, 1)


def test_sample_boundary_values():
    cfg = Config()
    x, t = sample_boundary(cfg)
    boundary_mask = (x == 0) | (x == 1)
    assert boundary_mask.all()


def test_sample_initial_size():
    cfg = Config()
    x, t = sample_initial(cfg)
    assert x.shape == (cfg.n_initial, 1)
    assert t.shape == (cfg.n_initial, 1)


def test_sample_initial_time():
    cfg = Config()
    x, t = sample_initial(cfg)
    assert (t == 0).all()
    assert (x >= 0).all() and (x <= 1).all()
