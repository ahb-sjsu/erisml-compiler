"""Arms of the precedent experiment (PREREG-precedent-v1-DRAFT.md, Section 4).

A case is (e, g, y, cluster): an encoder embedding, a tuple of gate bits read by the
extractor, a judgement, and the unit it was sampled with. A stratum is a gate pattern.

    G   stratum mean                       gates, no retrieval
    A   k nearest by cosine                retrieval, no gates
    M   k nearest under a learned metric   retrieval + metric, no gates
    B   A restricted to the query's stratum   (a precedent transfers only when its set of
                                               relevant differences on gate features is empty)
    Cp  B under one metric learned across strata (pooled)
    Cl  B under one metric per stratum (local)

The metric is a low-rank factor U, cost ||U(x - x')||, learned by metric learning for kernel
regression (leave-one-out Nadaraya-Watson loss; Weinberger and Tesauro 2007) on TRAINING cases
only, in coordinates discovered without labels (PCA of the training embeddings). Nothing here
reads a hand-made moral axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

K_NEIGHBOURS = 10  # fixed before any data; not tuned
PCA_DIM = 48
RANK = 8
MLKR_STEPS = 250
MLKR_LR = 0.05
MIN_STRATUM = 40  # fewer training cases than this: the local metric falls back to the pooled one
# Round 2 (2026-11-25): a metric fitted on one stratum alone could not be told from the pooled one
# in rehearsal (0 of 5). The local metric is now the pooled metric plus a penalised per-stratum
# part, U_s = U_0 + dU_s (Parameswaran and Weinberger 2010), started at U_0.
LOCAL_PENALTY = 0.01

ARMS = ("G", "A", "M", "B", "Cp", "Cl")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Cases:
    e: np.ndarray  # (n, d) unit-norm embeddings
    s: np.ndarray  # (n,) stratum id (gate pattern as an integer)
    y: np.ndarray  # (n,) judgement
    cluster: np.ndarray  # (n,) sampling unit


class Coords:
    """Label-free coordinates: PCA of the training embeddings, standardised."""

    def __init__(self, e_train: np.ndarray, dim: int = PCA_DIM):
        self.mu = e_train.mean(0)
        _, sv, vt = np.linalg.svd(e_train - self.mu, full_matrices=False)
        self.vt = vt[:dim]
        self.sd = (sv[:dim] / np.sqrt(len(e_train) - 1)) + 1e-12

    def __call__(self, e: np.ndarray) -> np.ndarray:
        return ((e - self.mu) @ self.vt.T) / self.sd


def fit_mlkr(
    z: np.ndarray,
    y: np.ndarray,
    same: np.ndarray | None,
    seed: int,
    anchor: np.ndarray | None = None,
    penalty: float = 0.0,
) -> np.ndarray:
    """U (RANK x dim) minimising the leave-one-out kernel-regression error of y.

    `same` (n x n bool) restricts each case's neighbours (None: all). y should already have
    whatever the gates explain removed, so the metric describes variation inside a stratum.
    """
    dev = device()
    g = torch.Generator(device="cpu").manual_seed(seed)
    zt = torch.tensor(z, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    n, d = zt.shape
    u0 = torch.zeros(RANK, d)
    u0[:, :RANK] = torch.eye(RANK)
    u0 = (u0 + 0.01 * torch.randn(RANK, d, generator=g)) / np.sqrt(2.0 * RANK)
    a0 = None
    if anchor is not None:
        a0 = torch.tensor(anchor, dtype=torch.float32, device=dev)
        u0 = a0.clone().cpu()
    u = u0.to(dev).requires_grad_(True)
    mask = torch.eye(n, dtype=torch.bool, device=dev)
    if same is not None:
        mask = mask | ~torch.tensor(same, device=dev)
    opt = torch.optim.Adam([u], lr=MLKR_LR)
    for _ in range(MLKR_STEPS):
        opt.zero_grad()
        p = zt @ u.T
        d2 = torch.cdist(p, p).pow(2).masked_fill(mask, float("inf"))
        w = torch.softmax(-d2, dim=1)
        w = torch.nan_to_num(w)  # a case alone in its stratum has no neighbours
        loss = ((yt - w @ yt) ** 2).mean()
        if a0 is not None and penalty > 0:
            loss = loss + penalty * ((u - a0) ** 2).sum()
        loss.backward()
        opt.step()
    return u.detach().cpu().numpy()


def _knn_mean(q: np.ndarray, x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Mean y of the k nearest rows of x to each row of q (Euclidean in the given space)."""
    dev = device()
    d2 = torch.cdist(
        torch.tensor(q, dtype=torch.float32, device=dev),
        torch.tensor(x, dtype=torch.float32, device=dev),
    )
    k = min(k, x.shape[0])
    idx = d2.topk(k, largest=False).indices.cpu().numpy()
    return y[idx].mean(1)


class Fitted:
    """Everything learned from the training cases. Queries never touch `fit`."""

    def __init__(self, train: Cases, seed: int = 0):
        self.train = train
        self.coords = Coords(train.e)
        self.z = self.coords(train.e)
        self.grand = float(train.y.mean())
        self.strata = np.unique(train.s)
        self.s_mean = {int(s): float(train.y[train.s == s].mean()) for s in self.strata}
        resid = train.y - np.array([self.s_mean[int(s)] for s in train.s])
        same = train.s[:, None] == train.s[None, :]
        self.u_nogate = fit_mlkr(self.z, train.y - self.grand, None, seed)
        self.u_pooled = fit_mlkr(self.z, resid, same, seed)
        self.u_local = {}
        for s in self.strata:
            m = train.s == s
            self.u_local[int(s)] = (
                fit_mlkr(self.z[m], resid[m], None, seed, anchor=self.u_pooled, penalty=LOCAL_PENALTY)
                if m.sum() >= MIN_STRATUM
                else self.u_pooled
            )

    def _within(self, q_space, x_space, q: Cases, fallback: np.ndarray) -> np.ndarray:
        out = fallback.copy()
        for s in np.unique(q.s):
            mq, mt = q.s == s, self.train.s == s
            if mt.sum() == 0:
                continue  # unseen stratum: the fallback stands
            out[mq] = _knn_mean(q_space(mq, int(s)), x_space(mt, int(s)), self.train.y[mt], K_NEIGHBOURS)
        return out

    def predict(self, q: Cases) -> dict[str, np.ndarray]:
        zq = self.coords(q.e)
        g = np.array([self.s_mean.get(int(s), self.grand) for s in q.s])
        out = {"G": g}
        # cosine on unit vectors is monotone in Euclidean distance on the raw embedding
        out["A"] = _knn_mean(q.e, self.train.e, self.train.y, K_NEIGHBOURS)
        out["M"] = _knn_mean(zq @ self.u_nogate.T, self.z @ self.u_nogate.T, self.train.y, K_NEIGHBOURS)
        out["B"] = self._within(lambda m, s: q.e[m], lambda m, s: self.train.e[m], q, g)
        out["Cp"] = self._within(
            lambda m, s: zq[m] @ self.u_pooled.T, lambda m, s: self.z[m] @ self.u_pooled.T, q, g
        )
        out["Cl"] = self._within(
            lambda m, s: zq[m] @ self.u_local[s].T, lambda m, s: self.z[m] @ self.u_local[s].T, q, g
        )
        return out

    def retrieved_ids(self, q_e: np.ndarray, k: int = K_NEIGHBOURS) -> np.ndarray:
        """Indices of the k nearest training cases under the gate-blind pooled metric."""
        dev = device()
        a = torch.tensor(self.coords(q_e) @ self.u_pooled.T, dtype=torch.float32, device=dev)
        b = torch.tensor(self.z @ self.u_pooled.T, dtype=torch.float32, device=dev)
        return torch.cdist(a, b).topk(k, largest=False).indices.cpu().numpy()
