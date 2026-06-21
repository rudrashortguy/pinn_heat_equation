class Config:
    alpha: float = 0.1
    n_collocation: int = 10000
    n_boundary: int = 200
    n_initial: int = 200
    n_hidden_layers: int = 5
    n_neurons: int = 50
    epochs_adam: int = 20000
    lr_adam: float = 0.001
    epochs_lbfgs: int = 1000
    lr_lbfgs: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
