# src/recruitment_fairness/models/fair_outcome_net.py
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------
# Gradient Reversal Layer
# ---------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GRL(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradReverse.apply(x, self.lambd)


# ---------------------------
# Plain (non-adversarial) MLP – kept for backward compatibility
# ---------------------------
class FairOutcomeNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # return logits (we'll use BCEWithLogitsLoss)
        return self.net(x)


def train_fair_outcome_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    n_epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    patience: int = 5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FairOutcomeNet(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = bce(logits, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = bce(val_logits, y_val).item()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[FairOutcomeNet] epoch={epoch:03d}  "
                f"train_loss={loss.item():.4f}  val_loss={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[FairOutcomeNet] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ---------------------------
# Adversarial (fairness) MLP
# ---------------------------
class FairOutcomeAdvNet(nn.Module):
    """
    Shared encoder -> (1) outcome head   (2) GRL -> adversarial sensitive-attribute head
    """

    def __init__(
        self,
        input_dim: int,
        n_groups: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        lambda_adv: float = 0.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.outcome_head = nn.Linear(hidden_dim, 1)  # logits
        self.grl = GRL(lambd=lambda_adv)
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_groups),  # logits
        )

    def forward(self, x):
        h = self.shared(x)
        y_logit = self.outcome_head(h)  # (N, 1)

        # pass through GRL so that encoder gradients get flipped
        h_grl = self.grl(h)
        g_logits = self.adv_head(h_grl)  # (N, n_groups)

        return y_logit, g_logits


@dataclass
class AdvTrainReturn:
    model: FairOutcomeAdvNet
    best_val_loss: float


def train_fair_outcome_net_adv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    g_val: np.ndarray,
    input_dim: int,
    n_groups: int,
    n_epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    lambda_adv: float = 0.1,
    patience: int = 5,
) -> FairOutcomeAdvNet:
    """
    Train adv net:
        loss = BCEWithLogits(outcome) + CrossEntropy(adv head)
    GRL internally flips gradients for the adversarial head, so we *add* the two losses.

    Args:
        g_*: integer group labels [0..n_groups-1]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FairOutcomeAdvNet(
        input_dim=input_dim,
        n_groups=n_groups,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_adv=lambda_adv,
    ).to(device)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    # tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    g_train = torch.tensor(g_train, dtype=torch.long, device=device)

    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)
    g_val = torch.tensor(g_val, dtype=torch.long, device=device)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        y_logit, g_logits = model(X_train)
        loss_out = bce(y_logit, y_train)
        loss_adv = ce(g_logits, g_train)
        loss = (
            loss_out + loss_adv
        )  # GRL already applies negative gradient through adv branch

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            yv_logit, gv_logits = model(X_val)
            val_loss = bce(yv_logit, y_val) + ce(gv_logits, g_val)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[FairOutcomeAdvNet] epoch={epoch:03d} "
                f"train_loss={loss.item():.4f} (out={loss_out.item():.4f},"
                f" adv={loss_adv.item():.4f}) "
                f"val_loss={val_loss.item():.4f}"
            )

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[FairOutcomeAdvNet] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
