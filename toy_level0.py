"""
toy_level0.py — Level 0 diagnostic toy for the "compressible cross-layer
interaction" hypothesis (H1a–H1d), pure-quadratic version (no network).

WHY THIS EXISTS
---------------
On a real LLM, if the rank-r surrogate mispredicts joint loss you cannot tell
WHICH of four things broke:
    (a) the true effective rank is high,
    (b) Gauss-Newton dropped an important (indefinite) residual,
    (c) the parallel discrete solver is weak,
    (d) the 2nd-order Taylor expansion broke at low bit-width.
Here the "loss" IS a known quadratic, so every one of those is a knob you set
by hand. This file is a *diagnostic instrument*, not a smaller experiment.

    true_loss(delta) = c^T delta + 1/2 delta^T (D + A^T A + E) delta
        delta_i in {-1, 0, +1},  ||delta||_0 <= K

    D         : block-diagonal local curvature (per "layer"), PSD
    A (r*xM)  : the TRUE low-rank cross-layer structure (planted spectrum)
    E         : the residual GN drops -- symmetric, INDEFINITE, ||E||_2 = eps
                (proxy for sum_k r_k * grad^2 f_k that Gauss-Newton discards)

The surrogate (what the method actually optimizes) uses only (D + A_r^T A_r)
and DROPS E -- exactly mirroring the GN assumption of the proposal. So the gap
between surrogate and true loss isolates (rank-truncation error) vs
(GN/residual error), which on a real model are hopelessly entangled.

Author's note to self: knobs implement the five synthetic-data pitfalls from
the design discussion -- coupling strength, planted cooperative pairs, a
low-energy decision-flip direction, per-layer scale heterogeneity, and a
spectrum you *set as input* rather than read out.

NOTE: run this yourself. Nothing here auto-executes except under __main__,
and even that is just a demo harness you invoke manually.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

# GPU-first per workflow preference. Override DEVICE by hand if you want CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # float64 for Level 0: we want the *structure* clean,
#                        not fp noise masquerading as approximation error.


def set_seed(seed: int) -> torch.Generator:
    """Reproducible generator bound to DEVICE."""
    torch.manual_seed(seed)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------------
# Config: every knob lives here so you can sweep without touching logic.
# ----------------------------------------------------------------------------
@dataclass
class ToyConfig:
    # --- geometry ---
    n_layers: int = 6
    cand_per_layer: int = 12          # m; M = n_layers * m
    budget_K: int = 8                 # ||delta||_0 <= K

    # --- KNOB 2: planted spectrum of the low-rank cross structure A ---
    r_true: int = 16                  # TRUE rank of A^T A (ground truth!)
    spectrum: str = "powerlaw"        # "powerlaw" | "exp" | "flat"
    alpha: float = 1.0                # power-law decay exponent (sigma_i ~ i^-alpha)
    exp_rate: float = 0.25            # decay rate if spectrum == "exp"
    sigma_scale: float = 1.0          # overall magnitude of singular values

    # --- KNOB 1: coupling strength (how much cross-term matters vs local D) ---
    coupling_lambda: float = 1.0      # multiplies A^T A contribution
    local_curv_scale: float = 1.0     # overall scale of block-diagonal D

    # --- KNOB 3: E = GN residual (indefinite, controlled norm & neg fraction) ---
    eps_norm: float = 0.0             # ||E||_2 ; set 0.0 to get a pure-GN world
    eps_rank: int = 24                # rank of E's factor form
    eps_neg_frac: float = 0.5         # fraction of E's eigenvalues that are < 0

    # --- KNOB 4: per-layer scale heterogeneity (S_l differs by layer) ---
    layer_scale_spread: float = 0.5   # 0 => all layers equal; >0 => heterogeneous

    # --- KNOB 5 (pitfall 2+3): plant pathological structure on purpose ---
    n_cooperative_pairs: int = 6      # pairs i,j: individually bad, jointly good
    plant_low_energy_decider: bool = True  # hide a decision-flip in a small sigma

    # --- linear term ---
    c_scale: float = 0.3

    # --- evaluation proposal pool (held-out joint proposals) ---
    n_eval_proposals: int = 4000
    eval_density: float = 0.5         # avg fraction of budget used per proposal

    seed: int = 0


# ----------------------------------------------------------------------------
# Problem container (everything in FACTORED form -- never materialize M x M).
# ----------------------------------------------------------------------------
@dataclass
class ToyProblem:
    cfg: ToyConfig
    M: int
    layer_of: Tensor                  # (M,) int, which layer each coord belongs to
    layer_slices: list[slice]
    c: Tensor                         # (M,)
    D_blocks: list[Tensor]            # per-layer dense PSD blocks (m x m)
    A: Tensor                         # (r_true, M)  -- true low-rank factor
    G: Tensor                         # (M, eps_rank) -- E factor
    E_eig: Tensor                     # (eps_rank,)  -- E eigenvalues (signed)
    scales: Tensor                    # (M,) per-coord weight-space scale S_l
    coop_pairs: list[tuple[int, int]] = field(default_factory=list)

    # ----- curvature actions (never form the full matrix) -----
    def quad(self, delta: Tensor) -> Tensor:
        """0.5 * delta^T (D + A^T A + E) delta, exact (includes E)."""
        return 0.5 * (self._dDd(delta) + self._aAa(delta) + self._eEe(delta))

    def true_loss(self, delta: Tensor) -> Tensor:
        return self.c @ delta + self.quad(delta)

    def _dDd(self, delta: Tensor) -> Tensor:
        out = delta.new_zeros(())
        for sl, Db in zip(self.layer_slices, self.D_blocks):
            d = delta[sl]
            out = out + d @ (Db @ d)
        return out

    def _aAa(self, delta: Tensor) -> Tensor:
        Ad = self.A @ delta                       # (r_true,)
        return self.cfg.coupling_lambda * (Ad @ Ad)

    def _eEe(self, delta: Tensor) -> Tensor:
        if self.G.numel() == 0:
            return delta.new_zeros(())
        Gd = self.G.T @ delta                     # (eps_rank,)
        return (Gd * self.E_eig) @ Gd

    def diag_curv(self) -> Tensor:
        """diag(D + lambda A^T A + E) -- used by the 'diagonal' baseline."""
        d = torch.zeros(self.M, device=DEVICE, dtype=DTYPE)
        for sl, Db in zip(self.layer_slices, self.D_blocks):
            d[sl] = torch.diagonal(Db)
        d = d + self.cfg.coupling_lambda * (self.A ** 2).sum(dim=0)
        if self.G.numel() > 0:
            d = d + (self.G ** 2 * self.E_eig.unsqueeze(0)).sum(dim=1)
        return d

    def full_curv_offdiag(self, i: int, j: int) -> Tensor:
        """M_ij of the TRUE curvature. For quadratic, pairwise interaction
        I_ij (second finite-difference) == M_ij exactly. Used to verify the
        planted cooperative pairs and the low-energy decider."""
        val = self.cfg.coupling_lambda * (self.A[:, i] @ self.A[:, j])
        if self.G.numel() > 0:
            val = val + (self.G[i] * self.E_eig) @ self.G[j]
        if self.layer_of[i] == self.layer_of[j]:
            li = self.layer_of[i].item()
            sl = self.layer_slices[li]
            Db = self.D_blocks[li]
            ii, jj = i - sl.start, j - sl.start
            val = val + Db[ii, jj]
        return val


# ----------------------------------------------------------------------------
# Spectrum builder (KNOB 2): you SET the singular profile, then verify the
# diagnostic reads it back -- the inverse of plotting an illustrative spectrum.
# ----------------------------------------------------------------------------
def _singular_values(cfg: ToyConfig) -> Tensor:
    r = cfg.r_true
    idx = torch.arange(1, r + 1, device=DEVICE, dtype=DTYPE)
    if cfg.spectrum == "powerlaw":
        s = idx ** (-cfg.alpha)
    elif cfg.spectrum == "exp":
        s = torch.exp(-cfg.exp_rate * (idx - 1))
    elif cfg.spectrum == "flat":
        s = torch.ones_like(idx)
    else:
        raise ValueError(f"unknown spectrum {cfg.spectrum!r}")
    return cfg.sigma_scale * s / s[0]             # normalized so sigma_1 == sigma_scale


def _random_orthonormal(rows: int, cols: int, g: torch.Generator) -> Tensor:
    """cols orthonormal columns in R^rows (cols <= rows)."""
    X = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE, generator=g)
    Q, _ = torch.linalg.qr(X)
    return Q[:, :cols]


# ----------------------------------------------------------------------------
# Problem generator
# ----------------------------------------------------------------------------
def make_problem(cfg: ToyConfig) -> ToyProblem:
    g = set_seed(cfg.seed)
    L, m = cfg.n_layers, cfg.cand_per_layer
    M = L * m
    layer_of = torch.arange(M, device=DEVICE) // m
    layer_slices = [slice(l * m, (l + 1) * m) for l in range(L)]

    # KNOB 4: per-layer scales S_l (weight-space step size differs by layer).
    base = torch.ones(L, device=DEVICE, dtype=DTYPE)
    jitter = torch.exp(cfg.layer_scale_spread *
                       torch.randn(L, device=DEVICE, dtype=DTYPE, generator=g))
    layer_scale = base * jitter
    scales = layer_scale[layer_of]                # (M,)

    # Block-diagonal D (local curvature), PSD, scaled per layer.
    D_blocks = []
    for l in range(L):
        root = torch.randn(m, m, device=DEVICE, dtype=DTYPE, generator=g) / math.sqrt(m)
        Db = root @ root.T + 0.1 * torch.eye(m, device=DEVICE, dtype=DTYPE)
        Db = cfg.local_curv_scale * (layer_scale[l] ** 2) * Db
        D_blocks.append(Db)

    # Low-rank A with planted spectrum. A = Sigma V^T, V^T has orthonormal rows.
    sig = _singular_values(cfg)                   # (r_true,)
    Vt = _random_orthonormal(M, cfg.r_true, g).T  # (r_true, M), orthonormal rows
    # fold per-coord scale into A so cross-interactions respect S_l too.
    A = (sig.unsqueeze(1) * Vt) * scales.unsqueeze(0)

    # KNOB 3: E = G diag(E_eig) G^T, indefinite, ||E||_2 = eps_norm.
    if cfg.eps_norm > 0.0 and cfg.eps_rank > 0:
        G = _random_orthonormal(M, cfg.eps_rank, g)           # (M, eps_rank)
        mag = torch.linspace(1.0, 0.3, cfg.eps_rank, device=DEVICE, dtype=DTYPE)
        n_neg = int(round(cfg.eps_neg_frac * cfg.eps_rank))
        sign = torch.ones(cfg.eps_rank, device=DEVICE, dtype=DTYPE)
        perm = torch.randperm(cfg.eps_rank, device=DEVICE, generator=g)
        sign[perm[:n_neg]] = -1.0
        E_eig = sign * mag
        E_eig = E_eig / E_eig.abs().max() * cfg.eps_norm       # ||E||_2 == eps_norm
    else:
        G = torch.zeros(M, 0, device=DEVICE, dtype=DTYPE)
        E_eig = torch.zeros(0, device=DEVICE, dtype=DTYPE)

    c = cfg.c_scale * torch.randn(M, device=DEVICE, dtype=DTYPE, generator=g) * scales

    prob = ToyProblem(cfg=cfg, M=M, layer_of=layer_of, layer_slices=layer_slices,
                      c=c, D_blocks=D_blocks, A=A, G=G, E_eig=E_eig, scales=scales)

    # KNOB 5 / pitfall 2: plant cooperative pairs (individually bad, jointly good).
    prob.coop_pairs = _plant_cooperative_pairs(prob, g)
    # KNOB 5 / pitfall 3: hide a decision-flip in a LOW-energy singular direction.
    if cfg.plant_low_energy_decider:
        _plant_low_energy_decider(prob, g)

    return prob


def _single_move_loss(prob: ToyProblem, i: int, v: float = 1.0) -> Tensor:
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    d[i] = v
    return prob.true_loss(d)


def _plant_cooperative_pairs(prob: ToyProblem, g: torch.Generator) -> list[tuple[int, int]]:
    """Create pairs (i,j) in DIFFERENT layers with dL_i>0, dL_j>0 but dL_ij<0.
    Mechanism: give i and j opposite loadings on one shared low-rank direction
    so (A^T A)_ij = a_i . a_j < 0 (strong negative cross), while c keeps each
    single move costly. Verified by assertion-style checks the caller can run."""
    cfg = prob.cfg
    L = cfg.n_layers
    pairs: list[tuple[int, int]] = []
    if cfg.r_true < 1:
        return pairs
    dir_k = 0  # reuse the top (highest-energy) direction -> easy to detect if needed
    for _ in range(cfg.n_cooperative_pairs):
        la, lb = torch.randperm(L, device=DEVICE, generator=g)[:2].tolist()
        i = la * cfg.cand_per_layer + int(torch.randint(cfg.cand_per_layer, (1,),
                                          device=DEVICE, generator=g))
        j = lb * cfg.cand_per_layer + int(torch.randint(cfg.cand_per_layer, (1,),
                                          device=DEVICE, generator=g))
        mag = 1.5 * cfg.sigma_scale
        prob.A[dir_k, i] = +mag * prob.scales[i]
        prob.A[dir_k, j] = -mag * prob.scales[j]       # opposite sign => negative cross
        # make each single move individually unattractive (positive dL).
        li = _single_move_loss(prob, i).item()
        lj = _single_move_loss(prob, j).item()
        if li < 0:
            prob.c[i] += (-li + 0.05)
        if lj < 0:
            prob.c[j] += (-lj + 0.05)
        pairs.append((i, j))
    return pairs


def _plant_low_energy_decider(prob: ToyProblem, g: torch.Generator) -> None:
    """Pitfall 3 made concrete: among a few near-tied candidates, inject a
    SMALL-sigma cross component that flips which combination is optimal.
    Truncating the surrogate below that rank keeps ~all spectral energy yet
    loses the decision -> demonstrates E(r) is not sufficient on its own."""
    cfg = prob.cfg
    if cfg.r_true < 2:
        return
    low_dir = cfg.r_true - 1                            # smallest-energy direction
    # pick a handful of coords spread across layers.
    n = min(4, cfg.n_layers)
    coords = [l * cfg.cand_per_layer + (l % cfg.cand_per_layer) for l in range(n)]
    small = prob.A[low_dir].abs().mean().clamp(min=1e-6)
    pattern = torch.tensor([+1.0, -1.0, +1.0, -1.0][:n], device=DEVICE, dtype=DTYPE)
    for idx, coord in enumerate(coords):
        prob.A[low_dir, coord] += 0.8 * small * pattern[idx] * prob.scales[coord]


# ----------------------------------------------------------------------------
# Surrogate factory: build A_r either from the ORACLE (truncate true A) or from
# a random-projection SKETCH. This is the Feasibility-A vs Feasibility-B split:
#   oracle  -> "is the structure even there?"  (upper bound)
#   sketch  -> "can we recover it cheaply?"    (realistic cost)
# ----------------------------------------------------------------------------
def oracle_lowrank(prob: ToyProblem, r: int) -> Tensor:
    """Best rank-r factor of the TRUE A^T A (truncated SVD of A)."""
    # A is (r_true, M); A^T A has rank <= r_true. Truncate via SVD of A.
    U, S, Vh = torch.linalg.svd(prob.A, full_matrices=False)  # A = U S Vh
    r = min(r, S.numel())
    # rank-r factor B with B^T B ~ truncated A^T A:  B = diag(S_r) Vh_r
    B = (S[:r].unsqueeze(1) * Vh[:r]) * math.sqrt(prob.cfg.coupling_lambda)
    return B                                                   # (r, M)


def sketch_lowrank(prob: ToyProblem, r: int, n_probes: int,
                   seed: int = 12345) -> tuple[Tensor, dict]:
    """Randomized sketch of the GN curvature H_GN = lambda * A^T A using
    matrix-vector products H_GN @ w = lambda * A^T (A w). Each product is the
    toy analogue of one JVP+VJP pair -- THIS is the cost Feasibility B is about.
    Returns (B_r of shape (r,M), info) where info['n_probes'] lets you plot
    recovery-vs-cost."""
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    lam = prob.cfg.coupling_lambda
    W = torch.randn(prob.M, n_probes, device=DEVICE, dtype=DTYPE, generator=g)
    Y = lam * (prob.A.T @ (prob.A @ W))                        # (M, n_probes)
    Q, _ = torch.linalg.qr(Y)                                  # range sketch
    # project and eigendecompose the small core (randomized eig for PSD).
    C = lam * (prob.A.T @ (prob.A @ Q))                        # (M, k)
    Bcore = Q.T @ C                                            # (k, k), ~symmetric
    Bcore = 0.5 * (Bcore + Bcore.T)
    evals, evecs = torch.linalg.eigh(Bcore)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp(min=0), evecs[:, order]
    r = min(r, evals.numel())
    factor = (Q @ evecs[:, :r]) * evals[:r].clamp(min=0).sqrt().unsqueeze(0)  # (M, r)
    B = factor.T                                               # (r, M)
    info = {"n_probes": n_probes, "recovered_sigma": evals[:r].sqrt().detach()}
    return B, info


def surrogate_loss(prob: ToyProblem, delta: Tensor, B: Tensor,
                   use_diag_only: bool = False) -> Tensor:
    """c^T d + 1/2 d^T D d + 1/2 ||B d||^2   (drops E, exactly like GN)."""
    lin = prob.c @ delta
    dDd = prob._dDd(delta)
    if use_diag_only:
        return lin + 0.5 * dDd
    Bd = B @ delta
    return lin + 0.5 * (dDd + Bd @ Bd)


# ----------------------------------------------------------------------------
# Discrete coordinate-descent primitive used by all solvers.
# Minimizes  g(d) = lin^T d + 1/2 d^T D d + 1/2 ||resid + F d||^2
# over d in {-1,0,+1}^m with ||d||_0 <= budget, maintaining F d incrementally.
# ----------------------------------------------------------------------------
def _coord_descent(lin: Tensor, D_block: Tensor, F: Tensor, resid: Tensor,
                   budget: int, sweeps: int = 4,
                   d0: Tensor | None = None) -> Tensor:
    m = lin.numel()
    d = torch.zeros(m, device=DEVICE, dtype=DTYPE) if d0 is None else d0.clone()
    a = resid + F @ d                      # running F d + resid   (len = rows of F)
    Dd = D_block @ d
    Fcol_sq = (F ** 2).sum(dim=0)          # ||F[:,i]||^2
    Dii = torch.diagonal(D_block)
    for _ in range(sweeps):
        changed = False
        for i in range(m):
            old = d[i].item()
            best_v, best_gain = old, 0.0
            nnz = int((d != 0).sum().item())
            for v in (-1.0, 0.0, 1.0):
                if v == old:
                    continue
                if old == 0.0 and v != 0.0 and nnz >= budget:
                    continue              # respect the L0 budget
                step = v - old
                # delta g for coord i, others fixed:
                dg = (lin[i].item() * step
                      + (Dd[i].item() * step + 0.5 * Dii[i].item() * step * step)
                      + ((F[:, i] @ a).item() * step
                         + 0.5 * Fcol_sq[i].item() * step * step))
                if dg < best_gain - 1e-12:
                    best_gain, best_v = dg, v
            if best_v != old:
                step = best_v - old
                a = a + F[:, i] * step
                Dd = Dd + D_block[:, i] * step
                d[i] = best_v
                changed = True
        if not changed:
            break
    return d


# ----------------------------------------------------------------------------
# Solvers (the five rows of the Experiment-3 comparison table).
# Each returns (delta, cost) where cost tracks the two axes H1d cares about.
# ----------------------------------------------------------------------------
@dataclass
class SolveCost:
    oracle_evals: int = 0       # true-loss ("full-model forward") evaluations
    syncs: int = 0              # shared-state refreshes (cross-layer comms)
    sketch_probes: int = 0      # state-construction cost (Feasibility B)


def _layer_budget(K: int, L: int) -> int:
    return max(1, K // L)


def solve_independent_parallel(prob: ToyProblem) -> tuple[Tensor, SolveCost]:
    """No cross info at all: each layer uses only c_l and D_l."""
    K = prob.cfg.budget_K
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    emptyF = torch.zeros(0, prob.cfg.cand_per_layer, device=DEVICE, dtype=DTYPE)
    for sl, Db in zip(prob.layer_slices, prob.D_blocks):
        resid = torch.zeros(0, device=DEVICE, dtype=DTYPE)
        d[sl] = _coord_descent(prob.c[sl], Db, emptyF, resid,
                               budget=_layer_budget(K, prob.cfg.n_layers))
    return _enforce_budget(prob, d), SolveCost(oracle_evals=1)


def solve_parallel_diagonal(prob: ToyProblem) -> tuple[Tensor, SolveCost]:
    """Global linear c + diagonal of full curvature, but NO cross terms."""
    K = prob.cfg.budget_K
    diag = prob.diag_curv()
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    emptyF = torch.zeros(0, prob.cfg.cand_per_layer, device=DEVICE, dtype=DTYPE)
    for sl in prob.layer_slices:
        Db = torch.diag(diag[sl])
        resid = torch.zeros(0, device=DEVICE, dtype=DTYPE)
        d[sl] = _coord_descent(prob.c[sl], Db, emptyF, resid,
                               budget=_layer_budget(K, prob.cfg.n_layers))
    return _enforce_budget(prob, d), SolveCost(oracle_evals=1)


def solve_parallel_lowrank(prob: ToyProblem, B: Tensor, outer_iters: int = 8,
                           trust_region: bool = False) -> tuple[Tensor, SolveCost]:
    """Shared-state parallel discrete refinement (proposal section 5).
    Shared state u = B delta. Each layer solves its local subproblem against
    u with its own block B_l; then a joint validation + (optional) discrete
    trust region + true-loss acceptance. B may be oracle or sketched."""
    K, L = prob.cfg.budget_K, prob.cfg.n_layers
    lb = _layer_budget(K, L)
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    cost = SolveCost()
    best = d.clone()
    best_true = prob.true_loss(best); cost.oracle_evals += 1

    for _ in range(outer_iters):
        u = B @ d                                   # shared global state
        cost.syncs += 1
        prop = d.clone()
        for li, sl in enumerate(prob.layer_slices):
            Bl = B[:, sl]
            resid = u - Bl @ d[sl]                  # u minus this layer's own part
            prop[sl] = _coord_descent(prob.c[sl], prob.D_blocks[li], Bl, resid,
                                      budget=lb, d0=d[sl])
        prop = _enforce_budget(prob, prop)

        # joint validation on the SURROGATE first (cheap).
        if trust_region:
            prop = _trust_region_shrink(prob, B, d, prop)

        # global acceptance on the TRUE loss (the expensive forward).
        t = prob.true_loss(prop); cost.oracle_evals += 1
        if t < best_true - 1e-10:
            best_true, best, d = t, prop.clone(), prop.clone()
        else:
            # no improvement: contract budget and try once more next iter.
            lb = max(1, lb - 1)
            d = best.clone()
    return best, cost


def _trust_region_shrink(prob: ToyProblem, B: Tensor, d_cur: Tensor,
                         prop: Tensor) -> Tensor:
    """Discrete trust region: if the joint proposal doesn't improve the
    surrogate, accept only the subset of coord-changes with the best marginal
    surrogate gain. No continuous learning rate -- must stay on the grid."""
    base = surrogate_loss(prob, d_cur, B)
    full = surrogate_loss(prob, prop, B)
    if full < base - 1e-12:
        return prop
    changes = (prop != d_cur).nonzero(as_tuple=True)[0]
    if changes.numel() == 0:
        return d_cur
    gains = []
    for idx in changes.tolist():
        trial = d_cur.clone(); trial[idx] = prop[idx]
        gains.append((surrogate_loss(prob, trial, B).item() - base.item(), idx))
    gains.sort()                                    # most-negative (best) first
    accepted = d_cur.clone()
    cur = base.item()
    for g_val, idx in gains:
        trial = accepted.clone(); trial[idx] = prop[idx]
        sl = surrogate_loss(prob, trial, B).item()
        if sl < cur - 1e-12 and int((trial != 0).sum()) <= prob.cfg.budget_K:
            accepted, cur = trial, sl
    return accepted


def solve_sequential_global(prob: ToyProblem) -> tuple[Tensor, SolveCost]:
    """Gold reference: greedy coordinate descent over ALL coords using the
    TRUE curvature (A AND E), updating global state after every single move.
    Strong but expensive -- the thing parallel methods must approach."""
    K = prob.cfg.budget_K
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    lam = prob.cfg.coupling_lambda
    aA = prob.A @ d
    aE = prob.G.T @ d if prob.G.numel() > 0 else torch.zeros(0, device=DEVICE, dtype=DTYPE)
    cost = SolveCost()
    Acol_sq = lam * (prob.A ** 2).sum(dim=0)
    for _ in range(K * 3):                           # enough sweeps for K picks
        best_dg, best_i, best_v = -1e-12, -1, 0.0
        nnz = int((d != 0).sum().item())
        for i in range(prob.M):
            old = d[i].item()
            li = prob.layer_of[i].item()
            sl = prob.layer_slices[li]
            Db = prob.D_blocks[li]
            ii = i - sl.start
            for v in (-1.0, 0.0, 1.0):
                if v == old:
                    continue
                if old == 0.0 and v != 0.0 and nnz >= K:
                    continue
                step = v - old
                # local D part
                Dd_i = (Db[ii] @ d[sl]).item()
                dg = prob.c[i].item() * step + Dd_i * step + 0.5 * Db[ii, ii].item() * step * step
                # low-rank cross part
                dg += lam * (prob.A[:, i] @ aA).item() * step + 0.5 * Acol_sq[i].item() * step * step
                # residual E part (seen only by the gold solver)
                if prob.G.numel() > 0:
                    Ei = (prob.G[i] * prob.E_eig) @ aE
                    dg += Ei.item() * step + 0.5 * ((prob.G[i] ** 2) * prob.E_eig).sum().item() * step * step
                if dg < best_dg:
                    best_dg, best_i, best_v = dg, i, v
        if best_i < 0:
            break
        step = best_v - d[best_i]
        aA = aA + prob.A[:, best_i] * step
        if prob.G.numel() > 0:
            aE = aE + prob.G[best_i] * step
        d[best_i] = best_v
        cost.oracle_evals += 1                       # each accepted move ~ a re-eval
        cost.syncs += 1
    return d, cost


def solve_exhaustive(prob: ToyProblem) -> tuple[Tensor, float]:
    """Brute-force global optimum. ONLY for tiny M (guarded). Ground-truth
    optimality gap for the heuristics."""
    M, K = prob.M, prob.cfg.budget_K
    if M > 18:
        raise RuntimeError(f"exhaustive infeasible for M={M}; use M<=18")
    best_d, best_l = torch.zeros(M, device=DEVICE, dtype=DTYPE), float("inf")
    coords = range(M)
    for k in range(K + 1):
        for support in itertools.combinations(coords, k):
            for signs in itertools.product((-1.0, 1.0), repeat=k):
                d = torch.zeros(M, device=DEVICE, dtype=DTYPE)
                for idx, s in zip(support, signs):
                    d[idx] = s
                l = prob.true_loss(d).item()
                if l < best_l:
                    best_l, best_d = l, d.clone()
    return best_d, best_l


def _enforce_budget(prob: ToyProblem, d: Tensor) -> Tensor:
    """If a parallel merge exceeds K, keep the coords with best marginal true gain."""
    nnz = int((d != 0).sum().item())
    if nnz <= prob.cfg.budget_K:
        return d
    idxs = (d != 0).nonzero(as_tuple=True)[0]
    base = prob.true_loss(torch.zeros_like(d)).item()
    scored = []
    for i in idxs.tolist():
        t = torch.zeros_like(d); t[i] = d[i]
        scored.append((prob.true_loss(t).item() - base, i))
    scored.sort()
    keep = set(i for _, i in scored[:prob.cfg.budget_K])
    out = torch.zeros_like(d)
    for i in keep:
        out[i] = d[i]
    return out


# ----------------------------------------------------------------------------
# Metrics -> map onto H1a, H1b, H1c, H1d.
# ----------------------------------------------------------------------------
def _spearman(x: Tensor, y: Tensor) -> float:
    rx = x.argsort().argsort().to(DTYPE)
    ry = y.argsort().argsort().to(DTYPE)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp(min=1e-12)
    return float((rx @ ry) / denom)


def random_proposals(prob: ToyProblem, n: int, seed: int = 999) -> Tensor:
    """Held-out joint proposals: random on-grid, budget-respecting."""
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    K = prob.cfg.budget_K
    out = torch.zeros(n, prob.M, device=DEVICE, dtype=DTYPE)
    for t in range(n):
        k = max(1, int(round(prob.cfg.eval_density * K)))
        supp = torch.randperm(prob.M, device=DEVICE, generator=g)[:k]
        signs = (torch.randint(0, 2, (k,), device=DEVICE, generator=g) * 2 - 1).to(DTYPE)
        out[t, supp] = signs
    return out


def h1a_interaction_significance(prob: ToyProblem) -> dict:
    """H1a: are cross-layer interactions big enough to flip discrete decisions?
    Uses exact I_ij = M_ij and the planted cooperative pairs."""
    # sample cross-layer pairs
    g = torch.Generator(device=DEVICE); g.manual_seed(7)
    n = 400
    I_vals, flips = [], 0
    for _ in range(n):
        i, j = torch.randint(prob.M, (2,), device=DEVICE, generator=g).tolist()
        if prob.layer_of[i] == prob.layer_of[j]:
            continue
        Iij = prob.full_curv_offdiag(i, j).item()
        I_vals.append(Iij)
        li, lj = _single_move_loss(prob, i).item(), _single_move_loss(prob, j).item()
        d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE); d[i] = 1; d[j] = 1
        lij = prob.true_loss(d).item()
        # decision flip: individuals say "don't", joint says "do" (or vice versa)
        indiv_good = (li < 0) or (lj < 0)
        joint_good = lij < min(0.0, li, lj)
        if joint_good and not indiv_good:
            flips += 1
    I_vals_t = torch.tensor(I_vals, device=DEVICE, dtype=DTYPE)
    # verify planted cooperative pairs really are cooperative
    coop_ok = 0
    for (i, j) in prob.coop_pairs:
        li, lj = _single_move_loss(prob, i).item(), _single_move_loss(prob, j).item()
        d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE); d[i] = 1; d[j] = 1
        lij = prob.true_loss(d).item()
        if li > 0 and lj > 0 and lij < 0:
            coop_ok += 1
    return {
        "I_abs_mean": float(I_vals_t.abs().mean()),
        "I_abs_p95": float(I_vals_t.abs().quantile(0.95)),
        "decision_flip_rate": flips / max(1, len(I_vals)),
        "cooperative_pairs_verified": f"{coop_ok}/{len(prob.coop_pairs)}",
    }


def h1b_compressibility(prob: ToyProblem, ranks: list[int], B_source: str = "oracle",
                        n_probes: int = 64) -> dict:
    """H1b: spectral energy E(r) AND decision-level metrics on held-out
    proposals. The whole point: E(r) can be high while ranking collapses."""
    # true singular values of A (energy of the planted cross structure)
    S = torch.linalg.svdvals(prob.A)
    energy = (S ** 2)
    Er = {r: float(energy[:r].sum() / energy.sum()) for r in ranks}

    props = random_proposals(prob, prob.cfg.n_eval_proposals)
    true_dL = torch.stack([prob.true_loss(props[t]) for t in range(props.shape[0])])

    out = {"E(r)": Er, "per_rank": {}}
    for r in ranks:
        B = (oracle_lowrank(prob, r) if B_source == "oracle"
             else sketch_lowrank(prob, r, n_probes)[0])
        sur = torch.stack([surrogate_loss(prob, props[t], B)
                           for t in range(props.shape[0])])
        # cross-layer-only interaction error: compare surrogate cross vs true cross
        cross_err = _cross_interaction_error(prob, B, props)
        k = max(1, prob.cfg.n_eval_proposals // 20)
        true_topk = set(true_dL.argsort()[:k].tolist())
        sur_topk = set(sur.argsort()[:k].tolist())
        out["per_rank"][r] = {
            "spearman": _spearman(sur, true_dL),
            "pearson": float(torch.corrcoef(torch.stack([sur, true_dL]))[0, 1]),
            "topk_overlap": len(true_topk & sur_topk) / k,
            "cross_interaction_rel_err": cross_err,
        }
    return out


def _cross_interaction_error(prob: ToyProblem, B: Tensor, props: Tensor) -> float:
    """Relative error of the surrogate on the CROSS-LAYER part only.
    true cross = 1/2 sum_{l<k} d_l^T M_lk d_k ; surrogate cross uses B.
    (This is the metric that catches a sketch that only learned intra-layer
    curvature.)"""
    def cross_true(d):
        lam = prob.cfg.coupling_lambda
        Ad = prob.A @ d
        full_quad_cross = lam * (Ad @ Ad)
        if prob.G.numel() > 0:
            Gd = prob.G.T @ d
            full_quad_cross = full_quad_cross + (Gd * prob.E_eig) @ Gd
        # subtract intra-layer (block-diagonal of A^T A + E)
        intra = d.new_zeros(())
        for sl in prob.layer_slices:
            dl = d[sl]
            Al = prob.A[:, sl]
            intra = intra + lam * (Al @ dl) @ (Al @ dl)
            if prob.G.numel() > 0:
                Gl = prob.G[sl]
                Gld = Gl.T @ dl
                intra = intra + (Gld * prob.E_eig) @ Gld
        return 0.5 * (full_quad_cross - intra)

    def cross_sur(d):
        Bd_full = B @ d
        full = Bd_full @ Bd_full
        intra = d.new_zeros(())
        for sl in prob.layer_slices:
            Bl = B[:, sl]
            bd = Bl @ d[sl]
            intra = intra + bd @ bd
        return 0.5 * (full - intra)

    num, den = 0.0, 0.0
    for t in range(min(500, props.shape[0])):
        ct = cross_true(props[t]).item()
        cs = cross_sur(props[t]).item()
        num += (ct - cs) ** 2
        den += ct ** 2
    return math.sqrt(num / max(den, 1e-12))


def h1c_coordination(prob: ToyProblem, B_oracle: Tensor, B_sketch: Tensor) -> dict:
    """H1c: does the shared state beat independent/diagonal parallel and
    approach sequential? Lower true loss is better."""
    results = {}
    d, _ = solve_independent_parallel(prob)
    results["independent_parallel"] = prob.true_loss(d).item()
    d, _ = solve_parallel_diagonal(prob)
    results["parallel_diagonal"] = prob.true_loss(d).item()
    d, _ = solve_parallel_lowrank(prob, B_oracle, trust_region=False)
    results["parallel_lowrank_oracle"] = prob.true_loss(d).item()
    d, _ = solve_parallel_lowrank(prob, B_oracle, trust_region=True)
    results["parallel_lowrank_oracle_TR"] = prob.true_loss(d).item()
    d, _ = solve_parallel_lowrank(prob, B_sketch, trust_region=True)
    results["parallel_lowrank_sketch_TR"] = prob.true_loss(d).item()
    d, _ = solve_sequential_global(prob)
    results["sequential_global"] = prob.true_loss(d).item()
    if prob.M <= 18:
        _, l = solve_exhaustive(prob)
        results["exhaustive_optimum"] = l
    return results


def h1d_cost(prob: ToyProblem, B_oracle: Tensor, B_sketch: Tensor,
             n_probes: int) -> dict:
    """H1d: cost on both axes -- oracle (true-loss) evals AND state-construction
    probes. A method can win on loss yet lose here if state-build dominates."""
    _, c_seq = solve_sequential_global(prob)
    _, c_par = solve_parallel_lowrank(prob, B_oracle, trust_region=True)
    return {
        "sequential": {"oracle_evals": c_seq.oracle_evals, "syncs": c_seq.syncs,
                       "sketch_probes": 0},
        "parallel_lowrank_oracle": {"oracle_evals": c_par.oracle_evals,
                                    "syncs": c_par.syncs, "sketch_probes": 0},
        "parallel_lowrank_sketch": {"oracle_evals": c_par.oracle_evals,
                                    "syncs": c_par.syncs,
                                    "sketch_probes": n_probes},
    }


# ----------------------------------------------------------------------------
# Demo harness. You run this; it does not run on import.
# ----------------------------------------------------------------------------
def run_once(cfg: ToyConfig, ranks: list[int] | None = None,
             n_probes: int = 64) -> dict:
    ranks = ranks or [2, 4, 8, 16, 24]
    prob = make_problem(cfg)
    B_oracle = oracle_lowrank(prob, max(ranks))
    B_sketch, sk_info = sketch_lowrank(prob, max(ranks), n_probes)
    report = {
        "config": cfg,
        "M": prob.M,
        "H1a_interaction": h1a_interaction_significance(prob),
        "H1b_oracle": h1b_compressibility(prob, ranks, "oracle"),
        "H1b_sketch": h1b_compressibility(prob, ranks, "sketch", n_probes),
        "H1c_coordination": h1c_coordination(prob, B_oracle, B_sketch),
        "H1d_cost": h1d_cost(prob, B_oracle, B_sketch, n_probes),
    }
    return report


def _pretty(report: dict) -> None:
    print(f"\n=== M = {report['M']} | device = {DEVICE} ===")
    print("\n[H1a] interaction significance:")
    for k, v in report["H1a_interaction"].items():
        print(f"   {k:<28}: {v}")
    print("\n[H1b] compressibility (ORACLE low-rank):")
    _pretty_h1b(report["H1b_oracle"])
    print("\n[H1b] compressibility (SKETCHED low-rank, Feasibility B):")
    _pretty_h1b(report["H1b_sketch"])
    print("\n[H1c] coordination (true loss, lower = better):")
    for k, v in report["H1c_coordination"].items():
        print(f"   {k:<32}: {v:+.6f}")
    print("\n[H1d] cost (two axes):")
    for k, v in report["H1d_cost"].items():
        print(f"   {k:<28}: {v}")


def _pretty_h1b(h1b: dict) -> None:
    print(f"   {'rank':>5} | {'E(r)':>7} | {'spearman':>8} | "
          f"{'topK_ovl':>8} | {'cross_relerr':>12}")
    for r, m in h1b["per_rank"].items():
        er = h1b["E(r)"][r]
        print(f"   {r:>5} | {er:>7.4f} | {m['spearman']:>8.4f} | "
              f"{m['topk_overlap']:>8.4f} | {m['cross_interaction_rel_err']:>12.4f}")


if __name__ == "__main__":
    # --- scenario A: pure-GN world (eps=0). If the method can't win HERE, stop. ---
    cfg_A = ToyConfig(spectrum="powerlaw", alpha=1.0, eps_norm=0.0, seed=0)
    _pretty(run_once(cfg_A))

    # --- scenario B: add indefinite residual (GN's blind spot). Watch spearman
    #     drop even while E(r) stays high -> that's GN-error, not rank-error. ---
    cfg_B = ToyConfig(spectrum="powerlaw", alpha=1.0, eps_norm=0.5,
                      eps_neg_frac=0.5, seed=0)
    _pretty(run_once(cfg_B))

    # --- scenario C: slow spectrum (hard to compress). Sanity that diagnostics
    #     actually REPORT incompressibility instead of hiding it. ---
    cfg_C = ToyConfig(spectrum="flat", eps_norm=0.0, seed=0)
    _pretty(run_once(cfg_C))

    # --- tiny instance to compare against the brute-force optimum ---
    cfg_tiny = ToyConfig(n_layers=3, cand_per_layer=5, budget_K=4, r_true=4,
                         eps_norm=0.3, n_eval_proposals=800, seed=1)
    _pretty(run_once(cfg_tiny, ranks=[1, 2, 3, 4]))
