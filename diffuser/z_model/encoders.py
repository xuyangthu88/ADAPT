import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, n_layers=2, act=nn.SiLU):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(dim, hidden), act()]
            dim = hidden
        layers += [nn.Linear(dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==============================
# Encoders for q(z|x,y) and p(z|x)
# ==============================

class EncoderQ(nn.Module):
    """Posterior encoder q(z|x,y) -> Gaussian parameters (mu, logvar)."""
    def __init__(self, x_dim, y_dim, z_dim, hidden=128):
        super().__init__()
        self.backbone = MLP(x_dim + y_dim, hidden, hidden=hidden, n_layers=3)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, xy):
        # h = self.backbone(torch.cat([x, y], dim=-1))
        h = self.backbone(xy) # state+obs一起输入
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        return mu, logvar


class EncoderP(nn.Module):
    """Prior encoder p(z|x) -> Gaussian parameters (mu, logvar)."""
    def __init__(self, x_dim, z_dim, hidden=128):
        super().__init__()
        self.backbone = MLP(x_dim, hidden, hidden=hidden, n_layers=3)
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        return mu, logvar