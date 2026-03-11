"""
Train joint world models on population trajectory data.

Architecture: one MLP per population.
  Input:  obs_self (10) + obs_partner (10) + action_self (1, embedded) = 21
  Output: next_obs_self (10)
  Loss:   MSE on next observation

Critically: no partner action in the input.
The model must implicitly learn what the partner will do in order to
accurately predict the ego agent's next observation (which includes
other_rel — the partner's relative position).

WM-A trained on Pop-A data encodes Pop-A conventions.
WM-B trained on Pop-B data encodes Pop-B conventions.
Same architecture, same loss, same data quantity.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

from coord_env import OBS_DIM, ACT_DIM

IN_DIM  = OBS_DIM * 2 + ACT_DIM  # obs_self + obs_partner + action onehot
OUT_DIM = OBS_DIM                  # next_obs_self


class WorldModel(nn.Module):
    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        layers = []
        in_d = IN_DIM
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, OUT_DIM))
        self.net = nn.Sequential(*layers)

        self.register_buffer("input_mean", torch.zeros(IN_DIM))
        self.register_buffer("input_std",  torch.ones(IN_DIM))
        self.register_buffer("output_mean", torch.zeros(OUT_DIM))
        self.register_buffer("output_std",  torch.ones(OUT_DIM))

    def forward(self, x):
        xn = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(xn) * (self.output_std + 1e-8) + self.output_mean

    def predict_np(self, obs_self: np.ndarray, obs_partner: np.ndarray,
                   action_self: int) -> np.ndarray:
        """Predict next_obs_self. Inputs: 10-dim numpy arrays."""
        device = next(self.parameters()).device
        act_onehot = np.zeros(ACT_DIM, dtype=np.float32)
        act_onehot[action_self] = 1.0
        x = np.concatenate([obs_self, obs_partner, act_onehot])
        with torch.no_grad():
            xt = torch.FloatTensor(x).unsqueeze(0).to(device)
            return self.forward(xt).squeeze(0).cpu().numpy()


def build_input(obs_self, obs_partner, actions):
    """Build (N, IN_DIM) input array."""
    N = len(actions)
    act_onehot = np.zeros((N, ACT_DIM), dtype=np.float32)
    act_onehot[np.arange(N), actions] = 1.0
    return np.concatenate([obs_self, obs_partner, act_onehot], axis=1)


def train_world_model(
    obs_self, obs_partner, actions, next_obs_self,
    *,
    epochs=200,
    batch_size=512,
    lr=1e-3,
    val_fraction=0.1,
    patience=20,
    device="cpu",
    verbose=True,
):
    model = WorldModel().to(device)

    X = build_input(obs_self, obs_partner, actions).astype(np.float32)
    Y = next_obs_self.astype(np.float32)

    # Normalization
    model.input_mean  = torch.FloatTensor(X.mean(0)).to(device)
    model.input_std   = torch.FloatTensor(X.std(0)).to(device)
    model.output_mean = torch.FloatTensor(Y.mean(0)).to(device)
    model.output_std  = torch.FloatTensor(Y.std(0)).to(device)

    tX = torch.FloatTensor(X)
    tY = torch.FloatTensor(Y)
    ds = TensorDataset(tX, tY)
    val_size   = max(1, int(len(ds) * val_fraction))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= val_size

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        sched.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {"best_val_loss": best_val_loss, "history": history}


def save_world_model(model: WorldModel, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, str(path))


def load_world_model(path, device="cpu") -> WorldModel:
    ckpt  = torch.load(str(path), map_location=device, weights_only=False)
    model = WorldModel()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path("outputs/world_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    losses = {}
    for pop in ["A", "B"]:
        print(f"\n{'='*50}")
        print(f"Training world model for Population {pop}")

        data = np.load(f"outputs/trajectories/traj_{pop}.npz")
        obs_self      = data["obs_self"]
        obs_partner   = data["obs_partner"]
        actions       = data["actions"]
        next_obs_self = data["next_obs_self"]
        print(f"  Transitions: {len(actions)}")

        model, info = train_world_model(
            obs_self, obs_partner, actions, next_obs_self,
            epochs=args.epochs,
            device=args.device,
            verbose=True,
        )
        losses[pop] = info["best_val_loss"]

        save_world_model(model, out_dir / f"wm_{pop}.pt")
        print(f"  Best val loss: {info['best_val_loss']:.6f}")
        print(f"  Saved to outputs/world_models/wm_{pop}.pt")

    print(f"\n{'='*50}")
    print(f"Val loss — WM_A: {losses['A']:.6f}, WM_B: {losses['B']:.6f}")
    print(f"Loss gap: {abs(losses['A'] - losses['B']):.6f}")
    print("  Small gap = equal quality (good — no quality confound)")

    with open(out_dir / "wm_metadata.json", "w") as f:
        json.dump(losses, f, indent=2)
