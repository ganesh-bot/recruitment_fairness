# src/recruitment_fairness/models/recruitment_net.py

# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RecruitmentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def train_recruitment_net(
    X_train, y_train, X_val, y_val, input_dim, n_epochs=20, lr=1e-3
):
    model = RecruitmentNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            print(
                f"Epoch {epoch+1}: Train loss={loss.item():.4f}  "
                f"Val loss={val_loss.item():.4f}"
            )
    return model
