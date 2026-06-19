"""
learning.py — deep-learning stack for the grid-world.

KEY DESIGN (learned the hard way, see the chat): cloning raw key-presses does
NOT generalize -- a small net can't learn global pathfinding. So we split:

    * NAVIGATION  -> BFS. Exact, optimal, generalizes perfectly. Not learned.
    * DECISION    -> a small network. Given the state, which station/task next?
                     THIS is what your demonstrations teach. It generalizes
                     (val_acc ~0.97, solves unseen maps) and is where text
                     grounding will plug in later.

From a human play-trace we extract, at each step, "which station did you choose
to go to next" -> that (state -> choice) pair trains the decision net.

Pieces:
  bfs_to_station(world, pos)   navigation primitive -> (dist, path)
  decision_features(world)     network input (per-station: remain, dist, reachable)
  DecisionNet                  MLP: features -> station logits
  DemoRecorder                 turns one human trace into (features -> choice) pairs
  DemoStore                    persists demos; splits train/val BY MAP
  train(net, store, ...)       returns history; plot_history draws the curve
  NeuralRoom                   save / load / list named models in neural_room/
  run_policy(net, world)       decision-net + BFS executor (watch-the-bot mode)
"""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld import Tile, STATION_ORDER, CARDINAL, GridWorld

# neural_room/ always lives next to this file, not in the launch directory
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _in_root(*parts):
    return os.path.join(_ROOT, *parts)


# ---- action space (used by the executor and by play.py) ------------------- #
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT"]
ACTION_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
INTERACT = 4

N_STATIONS = len(STATION_ORDER)
N_FEATURES = 3 * N_STATIONS                      # per station: remain, dist, reachable


# ---- navigation primitive (BFS) ------------------------------------------- #
def bfs_to_station(world, station_pos):
    """Shortest path from agent to any FLOOR cell cardinally adjacent to the
    station. Returns (distance, [ (dr,dc), ... ]) or None if unreachable."""
    s = world.size
    start = world.agent
    targets = {(station_pos[0] + dr, station_pos[1] + dc) for dr, dc in CARDINAL
               if world.in_bounds(station_pos[0] + dr, station_pos[1] + dc)
               and world.grid[station_pos[0] + dr][station_pos[1] + dc] == Tile.FLOOR}
    if not targets:
        return None
    if start in targets:
        return 0, []
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur in targets:
            path = []
            node = cur
            while prev[node] is not None:
                pr, pc = prev[node]
                path.append((node[0] - pr, node[1] - pc))
                node = prev[node]
            return len(path), path[::-1]
        for dr, dc in CARDINAL:
            nx = (cur[0] + dr, cur[1] + dc)
            if (world.in_bounds(*nx) and nx not in prev
                    and world.grid[nx[0]][nx[1]] == Tile.FLOOR):
                prev[nx] = cur
                q.append(nx)
    return None


# ---- decision features ---------------------------------------------------- #
def decision_features(world):
    """Per station type: [remaining_count/3, bfs_distance/20, reachable&needed].
    BFS supplies the geometry the net shouldn't have to infer; the net only has
    to learn the *policy* over it (and, later, word->station grounding)."""
    f = []
    for st in STATION_ORDER:
        rem = sum(1 for t in world.tasks if t.station == st and not t.done)
        pos = world._find_station(st)
        res = bfs_to_station(world, pos) if pos else None
        dist = res[0] if res else 99
        f += [rem / 3.0, min(dist, 20) / 20.0, 1.0 if (res and rem > 0) else 0.0]
    return np.asarray(f, dtype=np.float32)


# ---- model ---------------------------------------------------------------- #
class DecisionNet(nn.Module):
    def __init__(self, n_in=N_FEATURES, n_out=N_STATIONS, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ---- turn one human trace into training pairs ----------------------------- #
class DemoRecorder:
    """Drive this from the play loop:
        rec = DemoRecorder(plan_text)
        rec.begin(world)                      # at episode start
        ... player moves ...
        on a successful interact at station S:
            rec.commit(world, S)              # records (decision_state -> S)
        rec.finalize(episode_id)              # -> demo dict (or None)
    """

    def __init__(self, plan_text=""):
        self.plan_text = plan_text
        self.pending = None                      # features at the current decision
        self.feats, self.choices = [], []

    def begin(self, world):
        self.pending = decision_features(world)

    def commit(self, world, station):
        if self.pending is None:
            self.pending = decision_features(world)
        self.feats.append(self.pending)
        self.choices.append(STATION_ORDER.index(station))
        self.pending = decision_features(world)   # baseline for the next choice

    def finalize(self, episode_id):
        if not self.choices:
            return None
        return {
            "episode_id": episode_id,
            "plan_text": self.plan_text,
            "feats": np.asarray(self.feats, dtype=np.float32),
            "choices": np.asarray(self.choices, dtype=np.int64),
        }


class DemoStore:
    """All demonstrations on disk. Splits train/val by whole episode (map)."""

    def __init__(self, path=None):
        self.path = path or _in_root("neural_room", "demonstrations.pkl")
        self.demos = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.demos = pickle.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.demos, f)

    def add(self, demo):
        if demo:
            self.demos.append(demo)
            self.save()

    @property
    def n_episodes(self):
        return len(self.demos)

    @property
    def n_pairs(self):
        return sum(len(d["choices"]) for d in self.demos)

    def _stack(self, demos):
        X = np.concatenate([d["feats"] for d in demos])
        Y = np.concatenate([d["choices"] for d in demos])
        return torch.from_numpy(X), torch.from_numpy(Y)

    def build(self, val_frac=0.25, seed=0):
        idx = list(range(len(self.demos)))
        random.Random(seed).shuffle(idx)
        n_val = int(len(idx) * val_frac)
        if len(self.demos) < 4 or n_val == 0:
            return self._stack(self.demos), None
        val = [self.demos[i] for i in idx[:n_val]]
        train = [self.demos[i] for i in idx[n_val:]]
        return self._stack(train), self._stack(val)


# ---- training -------------------------------------------------------------- #
def train(net, store, epochs=120, lr=1e-2, wd=1e-4, batch=64, seed=0, log=print):
    (Xtr, Ytr), val = store.build(seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.size(0)
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}

    for ep in range(epochs):
        net.train()
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = order[i:i + batch]
            loss = F.cross_entropy(net(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        hist["train_loss"].append(tot / n)

        if val is not None:
            Xv, Yv = val
            net.eval()
            with torch.no_grad():
                logits = net(Xv)
                hist["val_loss"].append(F.cross_entropy(logits, Yv).item())
                hist["val_acc"].append((logits.argmax(1) == Yv).float().mean().item())

        if (ep + 1) % max(1, epochs // 6) == 0:
            m = f"epoch {ep+1:3d}  train_loss {hist['train_loss'][-1]:.3f}"
            if val is not None:
                m += f"  val_loss {hist['val_loss'][-1]:.3f}  val_acc {hist['val_acc'][-1]:.2f}"
            log(m)
    return hist


def plot_history(history, title="learning curve", save_path=None, show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], label="train loss", color="#c0504d")
    if history["val_loss"]:
        ax1.plot(ep, history["val_loss"], label="val loss", color="#e09246")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(loc="upper right")
    if history["val_acc"]:
        ax2 = ax1.twinx()
        ax2.plot(ep, history["val_acc"], "--", label="val acc", color="#5c9c78")
        ax2.set_ylabel("val accuracy (held-out maps)"); ax2.set_ylim(0, 1.02)
        ax2.legend(loc="lower right")
    ax1.set_title(title); fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# ---- neural room: named model registry ------------------------------------ #
class NeuralRoom:
    def __init__(self, root=None):
        self.root = root or _in_root("neural_room")
        self.reg_path = os.path.join(self.root, "registry.json")
        os.makedirs(self.root, exist_ok=True)
        self.registry = {}
        if os.path.exists(self.reg_path):
            with open(self.reg_path) as f:
                self.registry = json.load(f)

    def _flush(self):
        with open(self.reg_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def list_models(self):
        return sorted(self.registry.keys())

    def save(self, net, name, meta=None):
        torch.save(net.state_dict(), os.path.join(self.root, f"{name}.pt"))
        entry = self.registry.get(name, {"created": time.strftime("%Y-%m-%d %H:%M")})
        entry.update({"file": f"{name}.pt", "updated": time.strftime("%Y-%m-%d %H:%M")})
        if meta:
            entry.update(meta)
        self.registry[name] = entry
        self._flush()

    def load(self, name):
        net = DecisionNet()
        net.load_state_dict(torch.load(os.path.join(self.root, f"{name}.pt")))
        net.eval()
        return net, self.registry.get(name, {})


# ---- decision-net + BFS executor (watch-the-bot / "use an old one") -------- #
@torch.no_grad()
def run_policy(net, world, max_legs=12):
    """Let a trained model play: it picks a station, BFS walks there, interact."""
    net.eval()
    for _ in range(max_legs):
        if world.all_done():
            return True
        feats = torch.from_numpy(decision_features(world))[None]
        choice = int(net(feats).argmax(1))
        pos = world._find_station(STATION_ORDER[choice])
        res = bfs_to_station(world, pos) if pos else None
        if not res:
            return False
        for dr, dc in res[1]:
            world.move(dr, dc)
        if not world.interact():
            return False                          # picked a station with no task
    return world.all_done()
