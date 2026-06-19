"""
language.py — learn to map an instruction TEXT to an ordered plan of stations.

This is the fix for "the bot ignores what I type". The network here NEVER sees
the task list or the goal vector. Its only input is the tokenised instruction
text; its target is the ordered sequence of stations the expert visited. So the
ordering and the word->station grounding must be learned from language alone.

  tokenize / Vocab          text -> token ids
  TextDemoStore             stores {text, sequence}; splits train/val
  TextPlanNet               embedding -> GRU -> per-step station logits
  train_text(...)           returns history (val metric = exact-sequence match)
  plan_from_text(...)       text -> [station indices] (stops at STOP)
  run_text_policy(...)      execute that plan with BFS
  save_lang / load_lang     weights + vocab + config together
"""

import json
import os
import pickle
import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld import STATION_ORDER
from learning import _in_root, bfs_to_station

N_STATIONS = len(STATION_ORDER)
STOP = N_STATIONS                                # sequence-terminator class
N_OUT = N_STATIONS + 1
MAX_STEPS = 8                                    # max stations in a plan
MAXLEN = 40                                      # max instruction tokens

_word = re.compile(r"[a-z']+")


def tokenize(text):
    return _word.findall(text.lower())


class Vocab:
    def __init__(self):
        self.i2w = ["<pad>", "<unk>"]
        self.w2i = {"<pad>": 0, "<unk>": 1}

    def build(self, texts, min_count=1):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        for w, n in counts.items():
            if n >= min_count and w not in self.w2i:
                self.w2i[w] = len(self.i2w)
                self.i2w.append(w)

    def encode(self, text, maxlen=MAXLEN):
        ids = [self.w2i.get(tok, 1) for tok in tokenize(text)][:maxlen]
        return ids + [0] * (maxlen - len(ids))

    def to_list(self):
        return self.i2w

    @classmethod
    def from_list(cls, lst):
        v = cls()
        v.i2w = list(lst)
        v.w2i = {w: i for i, w in enumerate(v.i2w)}
        return v


class TextDemoStore:
    def __init__(self, path=None):
        self.path = path or _in_root("neural_room", "lang_demos.pkl")
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
    def n(self):
        return len(self.demos)

    def _pack(self, vocab, idx):
        X = torch.tensor([vocab.encode(self.demos[i]["text"]) for i in idx],
                         dtype=torch.long)
        Y = torch.full((len(idx), MAX_STEPS), STOP, dtype=torch.long)
        for row, i in enumerate(idx):
            for j, s in enumerate(self.demos[i]["sequence"][:MAX_STEPS]):
                Y[row, j] = s
        return X, Y

    def build(self, vocab, val_frac=0.2, seed=0):
        idx = list(range(len(self.demos)))
        random.Random(seed).shuffle(idx)
        nv = int(len(idx) * val_frac)
        if len(self.demos) < 5 or nv == 0:
            return self._pack(vocab, idx), None
        return self._pack(vocab, idx[nv:]), self._pack(vocab, idx[:nv])


START = N_OUT                                    # decoder start token
DEC_VOCAB = N_OUT + 1                            # stations + STOP + START


class TextPlanNet(nn.Module):
    """GRU encoder over the instruction -> GRU decoder emits the station plan
    one step at a time (autoregressive). Teacher-forced in training."""

    def __init__(self, vocab_size, emb=48, hid=96, dec_emb=24,
                 max_steps=MAX_STEPS, n_out=N_OUT):
        super().__init__()
        self.max_steps = max_steps
        self.n_out = n_out
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec_emb = nn.Embedding(DEC_VOCAB, dec_emb)
        self.dec = nn.GRU(dec_emb, hid, batch_first=True)
        self.out = nn.Linear(hid, n_out)

    def encode(self, x):
        _, h = self.enc(self.emb(x))
        return h

    def forward(self, x, dec_in):                 # teacher forcing
        o, _ = self.dec(self.dec_emb(dec_in), self.encode(x))
        return self.out(o)

    @torch.no_grad()
    def greedy(self, x, max_steps=None):
        max_steps = max_steps or self.max_steps
        h = self.encode(x)
        tok = torch.full((x.size(0), 1), START, dtype=torch.long)
        outs = []
        for _ in range(max_steps):
            o, h = self.dec(self.dec_emb(tok), h)
            tok = self.out(o[:, -1]).argmax(-1, keepdim=True)
            outs.append(tok)
        return torch.cat(outs, 1)


def train_text(net, store, vocab, epochs=120, lr=2e-3, wd=1e-4, batch=64,
               seed=0, log=print):
    import copy
    (Xtr, Ytr), val = store.build(vocab, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.size(0)
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    start_col = torch.full((n, 1), START, dtype=torch.long)
    dec_in_all = torch.cat([start_col, Ytr[:, :-1]], dim=1)   # shift right
    best_acc, best_state = -1.0, None

    def seq_match(pred, target):
        """Exact match up to (and including) the first STOP in the target."""
        ok = 0
        for p, t in zip(pred.tolist(), target.tolist()):
            tl = t[:t.index(STOP) + 1] if STOP in t else t
            pl = p[:p.index(STOP) + 1] if STOP in p else p
            ok += (pl == tl)
        return ok / len(target)

    for ep in range(epochs):
        net.train()
        order = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = order[i:i + batch]
            logits = net(Xtr[b], dec_in_all[b])
            loss = F.cross_entropy(logits.reshape(-1, net.n_out), Ytr[b].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        hist["train_loss"].append(tot / n)

        if val is not None:
            Xv, Yv = val
            net.eval()
            with torch.no_grad():
                di = torch.cat([torch.full((Xv.size(0), 1), START, dtype=torch.long),
                                Yv[:, :-1]], dim=1)
                hist["val_loss"].append(
                    F.cross_entropy(net(Xv, di).reshape(-1, net.n_out),
                                    Yv.reshape(-1)).item())
                acc = seq_match(net.greedy(Xv), Yv)
            hist["val_acc"].append(acc)
            if acc >= best_acc:                   # keep the best-generalizing model
                best_acc, best_state = acc, copy.deepcopy(net.state_dict())

        if (ep + 1) % max(1, epochs // 6) == 0:
            m = f"epoch {ep+1:3d}  train_loss {hist['train_loss'][-1]:.3f}"
            if val is not None:
                m += (f"  val_loss {hist['val_loss'][-1]:.3f}"
                      f"  val_seq_acc {hist['val_acc'][-1]:.2f}")
            log(m)

    if best_state is not None:                    # restore best checkpoint
        net.load_state_dict(best_state)
        log(f"restored best checkpoint (val_seq_acc {best_acc:.2f})")
    return hist


def plan_from_text(net, vocab, text):
    net.eval()
    x = torch.tensor([vocab.encode(text)], dtype=torch.long)
    pred = net.greedy(x)[0].tolist()
    plan = []
    for s in pred:
        if s == STOP:
            break
        plan.append(s)
    return plan


def run_text_policy(net, vocab, world, text, max_legs=MAX_STEPS):
    """Execute the text-derived plan: walk to each station with BFS, interact."""
    for s in plan_from_text(net, vocab, text)[:max_legs]:
        if world.all_done():
            break
        pos = world._find_station(STATION_ORDER[s])
        if pos is None:
            continue                              # text named an absent station
        res = bfs_to_station(world, pos)
        if not res:
            continue
        for dr, dc in res[1]:
            world.move(dr, dc)
        world.interact()
    return world.all_done()


# ---- persistence: weights + vocab + config together ----------------------- #
def save_lang(net, vocab, name, root=None):
    root = root or _in_root("neural_room")
    os.makedirs(root, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(root, f"{name}.lang.pt"))
    with open(os.path.join(root, f"{name}.lang.json"), "w") as f:
        json.dump({"vocab": vocab.to_list(),
                   "max_steps": net.max_steps, "n_out": net.n_out}, f)


def load_lang(name, root=None):
    root = root or _in_root("neural_room")
    with open(os.path.join(root, f"{name}.lang.json")) as f:
        cfg = json.load(f)
    vocab = Vocab.from_list(cfg["vocab"])
    net = TextPlanNet(len(vocab.i2w), max_steps=cfg["max_steps"], n_out=cfg["n_out"])
    net.load_state_dict(torch.load(os.path.join(root, f"{name}.lang.pt")))
    net.eval()
    return net, vocab
