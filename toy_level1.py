"""
toy_level1.py — Level 1 diagnostic: tiny network, REAL PTQ gap, REAL
Gauss-Newton curvature via autodiff, and the headline test the whole research
direction hinges on:

    does GN (which drops the residual sum_k r_k * grad^2 f_k) actually predict
    the TRUE cross-layer interaction I_ij -- or does the dropped residual, which
    is exactly non-zero here because PTQ leaves a teacher-student gap, silently
    corrupt the surrogate?

Level 0 could inject the residual E by hand. Level 1 cannot cheat: the residual
arises on its own from a nonzero teacher-student KL. So this is where the
concern "GN spectrum looks compressible while the surrogate mispredicts joint
loss" gets tested against a residual we did NOT design.

PIPELINE
--------
  teacher (FP)  :  fixed random MLP, defines targets via its softmax
  student       :  same weights, then quantized to a COARSE fixed grid
                   -> real, nonzero  L_PTQ = E_x KL(p_teacher || p_student) > 0
  candidates    :  a set of weight entries across layers; a +-1 integer move on
                   coordinate i shifts that weight by +- its (fixed) grid scale.
  global loss   :  teacher-student KL on a calibration set (the thing we refine)

WHAT IT MEASURES
----------------
  * c              = grad_delta L(0)                       (true linear term)
  * H_GN           = (F^{1/2} J)^T (F^{1/2} J)  exact      (Gauss-Newton on cands)
  * I_ij (true)    = L(+i+j)-L(+i)-L(+j)+L(0)  finite diff (REAL interaction)
  * residual       = how much H_GN misses I_ij  <-- the headline
  * sketch cost    = # backward passes to build A (Feasibility B, counted)
  * H1b on REAL L  = E(r), spearman, top-K, cross-err with TRUE loss as truth
  * H1c/H1d        = parallel-lowrank vs sequential, accepted on REAL forwards

Everything is in INTEGER-delta units, so c, H_GN and I_ij are directly
comparable (a +-1 move is the unit).

NOTE: run this yourself. torch.func (jacrev / functional_call) required.
Nothing executes on import; __main__ is a manual demo harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev, grad

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64   # structure must be clean; fp32 noise would masquerade
#                         as GN-residual, defeating the entire point of Level 1.


def set_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    return g


def _spearman(x: Tensor, y: Tensor) -> float:
    rx = x.argsort().argsort().to(DTYPE); ry = y.argsort().argsort().to(DTYPE)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (rx.norm() * ry.norm()).clamp(min=1e-12))


@dataclass
class SolveCost:
    real_forwards: int = 0     # true-loss ("full-model forward") evaluations
    syncs: int = 0             # shared-state refreshes
    sketch_backwards: int = 0  # state-construction cost (Feasibility B)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class Level1Config:
    in_dim: int = 24
    hidden: int = 32
    n_hidden_layers: int = 3           # => 4 weight matrices (incl. output)
    n_classes: int = 8
    n_calib: int = 96                  # calibration samples for the global loss
    n_heldout: int = 96                # separate split for honest prediction

    bits: int = 3                      # KNOB: fewer bits => bigger PTQ gap/residual
    per_channel: bool = False          # per-tensor (False) or per-output-channel

    cands_per_layer: int = 12          # candidates sampled per weight matrix
    budget_K: int = 10
    diversity_aware: bool = True       # spread candidates, include low-grad ones

    sketch_ranks: tuple[int, ...] = (2, 4, 8, 16, 32)
    sketch_oversample: int = 8         # extra probes beyond rank for randomized SVD

    n_interaction_pairs: int = 300     # cross-layer pairs for I_ij
    n_eval_proposals: int = 1500       # held-out joint proposals for H1b

    seed: int = 0


# ----------------------------------------------------------------------------
# Tiny network
# ----------------------------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self, cfg: Level1Config):
        super().__init__()
        dims = [cfg.in_dim] + [cfg.hidden] * cfg.n_hidden_layers + [cfg.n_classes]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        for k, lin in enumerate(self.layers):
            x = lin(x)
            if k < len(self.layers) - 1:
                x = self.act(x)
        return x                                       # logits (N, n_classes)


# ----------------------------------------------------------------------------
# Quantization to a fixed coarse grid  (this is what creates the real PTQ gap)
# ----------------------------------------------------------------------------
def quantize_weight(w: Tensor, bits: int, per_channel: bool) -> tuple[Tensor, Tensor]:
    """Symmetric fixed-grid quant. Returns (w_quant, scale). Scale & integer
    codes are then held FIXED; candidates only nudge codes by +-1."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:                                    # scale per output row
        s = w.abs().amax(dim=1, keepdim=True) / qmax
    else:
        s = w.abs().amax() / qmax
    s = s.clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -qmax - 1, qmax)
    return q * s, s


# ----------------------------------------------------------------------------
# Problem container
# ----------------------------------------------------------------------------
@dataclass
class L1Problem:
    cfg: Level1Config
    model: TinyMLP
    base_params: dict[str, Tensor]     # fixed quantized student params
    X_calib: Tensor
    X_heldout: Tensor
    teacher_logp: Tensor               # teacher log-probs on calib (fixed targets)
    teacher_logp_ho: Tensor
    # candidate bookkeeping
    cand_param: list[str]              # param name per candidate
    cand_flat_idx: Tensor             # (M,) flat index within that param
    cand_scale: Tensor                # (M,) grid scale for a +-1 move
    cand_layer: Tensor               # (M,) which layer (for block structure)
    M: int
    layer_slices: list[slice] = field(default_factory=list)

    # ------- differentiable perturbation of weights by integer deltas -------
    def _perturbed_params(self, delta: Tensor, names_idx: dict) -> dict:
        pert = {}
        for name, base in self.base_params.items():
            if name in names_idx:
                idx, mask = names_idx[name]
                vals = delta[mask] * self.cand_scale[mask]
                add = torch.zeros(base.numel(), device=DEVICE, dtype=DTYPE)
                add = add.index_add(0, idx, vals)
                pert[name] = base + add.view_as(base)
            else:
                pert[name] = base
        return pert

    def _names_idx(self) -> dict:
        d = {}
        for name in set(self.cand_param):
            mask = torch.tensor([p == name for p in self.cand_param], device=DEVICE)
            idx = self.cand_flat_idx[mask]
            d[name] = (idx, mask)
        return d

    def logits(self, delta: Tensor, heldout: bool = False) -> Tensor:
        X = self.X_heldout if heldout else self.X_calib
        pert = self._perturbed_params(delta, self._names_idx())
        return functional_call(self.model, pert, (X,))

    def loss(self, delta: Tensor, heldout: bool = False) -> Tensor:
        """E_x KL(teacher || student). Teacher log-probs are fixed targets."""
        tlp = self.teacher_logp_ho if heldout else self.teacher_logp
        logits = self.logits(delta, heldout)
        slp = torch.log_softmax(logits, dim=1)
        t = tlp.exp()
        return (t * (tlp - slp)).sum(dim=1).mean()

    def loss_value(self, delta: Tensor, heldout: bool = False) -> float:
        with torch.no_grad():
            return float(self.loss(delta, heldout))


def build_problem(cfg: Level1Config) -> L1Problem:
    g = set_seed(cfg.seed)
    teacher = TinyMLP(cfg).to(DEVICE, DTYPE)
    with torch.no_grad():
        for lin in teacher.layers:                     # fixed random teacher
            lin.weight.copy_(torch.randn_like(lin.weight) /
                             math.sqrt(lin.weight.shape[1]))
    student = TinyMLP(cfg).to(DEVICE, DTYPE)
    base_params, scales_by_name = {}, {}
    with torch.no_grad():
        for (tn, tp), (sn, sp) in zip(teacher.named_parameters(),
                                      student.named_parameters()):
            wq, s = quantize_weight(tp.data, cfg.bits, cfg.per_channel)
            sp.data.copy_(wq)
            base_params[sn] = wq.detach().clone()
            # expand per-tensor / per-channel scale to per-element for candidate use
            if cfg.per_channel:
                scales_by_name[sn] = s.expand_as(wq).reshape(-1).clone()
            else:
                scales_by_name[sn] = (s * torch.ones_like(wq)).reshape(-1).clone()

    X_calib = torch.randn(cfg.n_calib, cfg.in_dim, device=DEVICE, dtype=DTYPE, generator=g)
    X_ho = torch.randn(cfg.n_heldout, cfg.in_dim, device=DEVICE, dtype=DTYPE, generator=g)
    with torch.no_grad():
        teacher_logp = torch.log_softmax(teacher(X_calib), dim=1)
        teacher_logp_ho = torch.log_softmax(teacher(X_ho), dim=1)

    prob = L1Problem(
        cfg=cfg, model=student, base_params=base_params,
        X_calib=X_calib, X_heldout=X_ho,
        teacher_logp=teacher_logp, teacher_logp_ho=teacher_logp_ho,
        cand_param=[], cand_flat_idx=torch.empty(0), cand_scale=torch.empty(0),
        cand_layer=torch.empty(0), M=0,
    )
    _select_candidates(prob, scales_by_name, g)
    return prob


# ----------------------------------------------------------------------------
# Candidate selection (pitfall: don't pick top-K gradient only -> miss
# cooperative moves). Diversity-aware = spread across layers + mix high/low grad.
# ----------------------------------------------------------------------------
def _select_candidates(prob: L1Problem, scales_by_name: dict,
                       g: torch.Generator) -> None:
    cfg = prob.cfg
    names = [n for n, _ in prob.model.named_parameters()]
    # per-coordinate gradient magnitude of the loss at delta=0, for ranking.
    cand_param, cand_idx, cand_scale, cand_layer = [], [], [], []
    for layer_i, name in enumerate(names):
        base = prob.base_params[name]
        numel = base.numel()
        s = scales_by_name[name]
        # gradient of loss wrt THIS weight tensor at the current (quantized) point
        wd = base.detach().clone().requires_grad_(True)
        pert = {n: (wd.view_as(base) if n == name else prob.base_params[n])
                for n in prob.base_params}
        logits = functional_call(prob.model, pert, (prob.X_calib,))
        slp = torch.log_softmax(logits, dim=1)
        loss = (prob.teacher_logp.exp() * (prob.teacher_logp - slp)).sum(1).mean()
        gmag = torch.autograd.grad(loss, wd)[0].reshape(-1).abs()

        k = min(cfg.cands_per_layer, numel)
        if cfg.diversity_aware:
            n_top = k // 2
            top = torch.topk(gmag, n_top).indices
            rest = torch.randperm(numel, device=DEVICE, generator=g)
            rest = rest[~torch.isin(rest, top)][: k - n_top]
            chosen = torch.cat([top, rest])
        else:
            chosen = torch.topk(gmag, k).indices
        for idx in chosen.tolist():
            cand_param.append(name)
            cand_idx.append(idx)
            cand_scale.append(s[idx].item())
            cand_layer.append(layer_i)

    prob.cand_param = cand_param
    prob.cand_flat_idx = torch.tensor(cand_idx, device=DEVICE)
    prob.cand_scale = torch.tensor(cand_scale, device=DEVICE, dtype=DTYPE)
    prob.cand_layer = torch.tensor(cand_layer, device=DEVICE)
    prob.M = len(cand_param)
    # layer slices assume candidates grouped by layer in insertion order
    prob.layer_slices = []
    start = 0
    for layer_i in sorted(set(cand_layer)):
        cnt = sum(1 for l in cand_layer if l == layer_i)
        prob.layer_slices.append(slice(start, start + cnt))
        start += cnt


# ----------------------------------------------------------------------------
# Exact candidate-space quantities via autodiff.
# ----------------------------------------------------------------------------
def candidate_gradient(prob: L1Problem) -> Tensor:
    """c = grad_delta L(0)."""
    delta0 = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    return grad(lambda d: prob.loss(d))(delta0).detach()


def candidate_jacobian(prob: L1Problem) -> Tensor:
    """J_cand = d logits_flat / d delta  at delta=0, shape (N*d, M)."""
    delta0 = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    J = jacrev(lambda d: prob.logits(d).reshape(-1))(delta0)
    return J.detach()                                  # (N*d, M)


def fisher_half_apply(prob: L1Problem, J: Tensor) -> Tensor:
    """Return R = F^{1/2} J, where F is the per-sample softmax Fisher of the
    STUDENT at delta=0 (the correct pullback metric for KL). Block-diagonal
    over samples, each block (d x d)."""
    cfg = prob.cfg
    N, d = cfg.n_calib, cfg.n_classes
    with torch.no_grad():
        logits0 = prob.logits(torch.zeros(prob.M, device=DEVICE, dtype=DTYPE))
        p = torch.softmax(logits0, dim=1)              # (N, d)
    R = torch.empty_like(J)
    Jr = J.reshape(N, d, prob.M)
    for x in range(N):
        px = p[x]
        F = torch.diag(px) - px.outer(px)              # (d, d), PSD
        evals, evecs = torch.linalg.eigh(F)
        Fhalf = evecs @ torch.diag(evals.clamp(min=0).sqrt()) @ evecs.T
        R[x * d:(x + 1) * d] = Fhalf @ Jr[x]
    return R                                            # (N*d, M)


def exact_gn(prob: L1Problem) -> tuple[Tensor, Tensor]:
    """Return (H_GN, R). H_GN = R^T R exactly (no sketch)."""
    J = candidate_jacobian(prob)
    R = fisher_half_apply(prob, J)
    return R.T @ R, R


# ----------------------------------------------------------------------------
# Random output-Jacobian sketch (the proposal's standard estimator).
# A = (1/sqrt r) [ v_t^T F^{1/2} J ]_{t=1..r}. Each row = one VJP = one backward.
# We COUNT backwards so Feasibility B is measured, not assumed.
# ----------------------------------------------------------------------------
def sketch_gn(prob: L1Problem, r: int) -> tuple[Tensor, SolveCost]:
    """Returns low-rank factor B (r x M) with E[B^T B] = H_GN, plus cost.
    Implemented via randomized range-finding on R using VJPs through logits."""
    cfg = prob.cfg
    N, d = cfg.n_calib, cfg.n_classes
    n_probes = r + cfg.sketch_oversample
    cost = SolveCost()

    with torch.no_grad():
        logits0 = prob.logits(torch.zeros(prob.M, device=DEVICE, dtype=DTYPE))
        p = torch.softmax(logits0, dim=1)
        Fhalfs = []
        for x in range(N):
            px = p[x]
            F = torch.diag(px) - px.outer(px)
            ev, U = torch.linalg.eigh(F)
            Fhalfs.append(U @ torch.diag(ev.clamp(min=0).sqrt()) @ U.T)
        Fhalf = torch.stack(Fhalfs)                    # (N, d, d)

    gph = torch.Generator(device=DEVICE); gph.manual_seed(1234 + r)

    def Rt_apply(v_blocks: Tensor) -> Tensor:
        """Compute R^T v = J^T F^{1/2} v via one VJP through logits. Counts 1 bwd."""
        u = torch.einsum("nij,nj->ni", Fhalf, v_blocks).reshape(-1)  # F^{1/2} v
        delta = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = prob.logits(delta).reshape(-1)
        (gv,) = torch.autograd.grad(out, delta, grad_outputs=u)
        cost.sketch_backwards += 1
        return gv.detach()                             # (M,)

    # randomized range finding on R^T (M x (N*d)): probe with random v in output space
    Ycols = []
    for _ in range(n_probes):
        v = torch.randn(N, d, device=DEVICE, dtype=DTYPE, generator=gph)
        Ycols.append(Rt_apply(v))
    Y = torch.stack(Ycols, dim=1)                      # (M, n_probes) ~ range(R^T R)
    Q, _ = torch.linalg.qr(Y)                          # (M, n_probes)
    # project H_GN onto Q without forming it: need R Q then (RQ)^T(RQ). But R Q is a
    # JVP; approximate core via R^T R Q = sum of VJP(JVP). Cheaper here: use Y already
    # = H_GN @ Omega; form core = Q^T H_GN Q via extra VJPs on Q's columns.
    core_cols = []
    # H_GN @ q = R^T (R q); R q = F^{1/2} J q is a JVP. Do it as VJP-of-scalar trick:
    for cidx in range(Q.shape[1]):
        q = Q[:, cidx]
        # J q  (forward sensitivity) via grad of (logits . dummy) is awkward; instead
        # use finite-free identity: (R^T R) q obtained by VJP( F (J q) ). We get J q
        # through a JVP using double-backward:
        delta = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE, requires_grad=True)
        out = prob.logits(delta).reshape(-1)
        w = torch.zeros_like(out, requires_grad=True)
        g1 = torch.autograd.grad(out, delta, grad_outputs=w, create_graph=True)[0]
        Jq = torch.autograd.grad(g1 @ q, w)[0]          # = J q   (N*d,)
        cost.sketch_backwards += 1
        Jq_blocks = Jq.reshape(N, d)
        FJq = torch.einsum("nij,nj->ni", Fhalf, torch.einsum("nij,nj->ni", Fhalf, Jq_blocks))
        core_cols.append(Rt_apply_noFhalf(prob, FJq, cost))
    core = Q.T @ torch.stack(core_cols, dim=1)          # (k, k)
    core = 0.5 * (core + core.T)
    evals, evecs = torch.linalg.eigh(core)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp(min=0), evecs[:, order]
    r = min(r, evals.numel())
    factor = (Q @ evecs[:, :r]) * evals[:r].sqrt().unsqueeze(0)  # (M, r)
    return factor.T.detach(), cost


def Rt_apply_noFhalf(prob: L1Problem, u_blocks: Tensor, cost: SolveCost) -> Tensor:
    """J^T u via one VJP (u already includes any metric)."""
    u = u_blocks.reshape(-1)
    delta = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE, requires_grad=True)
    out = prob.logits(delta).reshape(-1)
    (gv,) = torch.autograd.grad(out, delta, grad_outputs=u)
    cost.sketch_backwards += 1
    return gv.detach()


# ----------------------------------------------------------------------------
# THE HEADLINE: real interaction vs GN prediction.
# I_ij (true) by finite difference of the REAL loss; compare to H_GN[i,j].
# ----------------------------------------------------------------------------
def interaction_audit(prob: L1Problem, H_GN: Tensor) -> dict:
    cfg = prob.cfg
    g = torch.Generator(device=DEVICE); g.manual_seed(42)
    L0 = prob.loss_value(torch.zeros(prob.M, device=DEVICE, dtype=DTYPE))

    def single(i, v=1.0):
        d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE); d[i] = v
        return prob.loss_value(d)

    cache = {}
    def single_c(i):
        if i not in cache:
            cache[i] = single(i, +1.0)
        return cache[i]

    true_I, gn_I, same_layer = [], [], []
    forwards = 1
    for _ in range(cfg.n_interaction_pairs):
        i, j = torch.randint(prob.M, (2,), device=DEVICE, generator=g).tolist()
        if i == j:
            continue
        d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE); d[i] = 1; d[j] = 1
        Lij = prob.loss_value(d); forwards += 1
        Li, Lj = single_c(i), single_c(j)
        Iij = Lij - Li - Lj + L0
        true_I.append(Iij)
        gn_I.append(H_GN[i, j].item())                 # GN predicts I_ij = H_ij
        same_layer.append(bool(prob.cand_layer[i] == prob.cand_layer[j]))

    forwards += len(cache)
    t = torch.tensor(true_I, device=DEVICE, dtype=DTYPE)
    h = torch.tensor(gn_I, device=DEVICE, dtype=DTYPE)
    sl = torch.tensor(same_layer, device=DEVICE)
    cross = ~sl

    def report(mask):
        if mask.sum() < 3:
            return {"n": int(mask.sum())}
        tt, hh = t[mask], h[mask]
        rel = (tt - hh).norm() / tt.norm().clamp(min=1e-12)
        pear = torch.corrcoef(torch.stack([tt, hh]))[0, 1]
        sign_disagree = ((tt.sign() != hh.sign()) & (tt.abs() > 1e-9)).float().mean()
        return {
            "n": int(mask.sum()),
            "pearson_true_vs_GN": float(pear),
            "residual_rel_L2": float(rel),        # <-- how much GN misses
            "sign_disagree_rate": float(sign_disagree),
            "true_I_abs_mean": float(tt.abs().mean()),
        }

    return {
        "L_PTQ(baseline KL)": L0,
        "all_pairs": report(torch.ones_like(sl)),
        "cross_layer_pairs": report(cross),       # the ones GN is supposed to model
        "same_layer_pairs": report(sl),
        "forwards_used": forwards,
    }


# ----------------------------------------------------------------------------
# H1b on the REAL loss: does the GN surrogate predict true joint dL?
# Separates (energy E(r)) from (decision quality) exactly as Level 0, but now
# ground truth is the real network loss, not a planted quadratic.
# ----------------------------------------------------------------------------
def random_proposals(prob: L1Problem, n: int, density: float = 0.5,
                     seed: int = 999) -> Tensor:
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    K = prob.cfg.budget_K
    out = torch.zeros(n, prob.M, device=DEVICE, dtype=DTYPE)
    for t in range(n):
        k = max(1, int(round(density * K)))
        supp = torch.randperm(prob.M, device=DEVICE, generator=g)[:k]
        signs = (torch.randint(0, 2, (k,), device=DEVICE, generator=g) * 2 - 1).to(DTYPE)
        out[t, supp] = signs
    return out


def surrogate_dL(prob: L1Problem, c: Tensor, B: Tensor, delta: Tensor) -> float:
    """GN surrogate predicted delta-loss: c^T d + 1/2 ||B d||^2  (drops residual)."""
    Bd = B @ delta
    return float(c @ delta + 0.5 * (Bd @ Bd))


def h1b_real(prob: L1Problem, c: Tensor, R: Tensor, ranks) -> dict:
    # true singular values of R => energy of the GN response space
    S = torch.linalg.svdvals(R)
    energy = S ** 2
    Er = {r: float(energy[:r].sum() / energy.sum()) for r in ranks}

    props = random_proposals(prob, prob.cfg.n_eval_proposals)
    L0 = prob.loss_value(torch.zeros(prob.M, device=DEVICE, dtype=DTYPE))
    true_dL = torch.tensor(
        [prob.loss_value(props[t]) - L0 for t in range(props.shape[0])],
        device=DEVICE, dtype=DTYPE)

    # oracle low-rank factor from truncated SVD of R: B_r = Sigma_r V_r^T
    U, Svd, Vh = torch.linalg.svd(R, full_matrices=False)
    out = {"E(r)": Er, "per_rank": {}}
    for r in ranks:
        r_ = min(r, Svd.numel())
        B = Svd[:r_].unsqueeze(1) * Vh[:r_]            # (r_, M), B^T B ~ trunc H_GN
        sur = torch.tensor([surrogate_dL(prob, c, B, props[t])
                            for t in range(props.shape[0])],
                           device=DEVICE, dtype=DTYPE)
        k = max(1, prob.cfg.n_eval_proposals // 20)
        tset = set(true_dL.argsort()[:k].tolist())
        sset = set(sur.argsort()[:k].tolist())
        out["per_rank"][r] = {
            "spearman": _spearman(sur, true_dL),
            "pearson": float(torch.corrcoef(torch.stack([sur, true_dL]))[0, 1]),
            "topk_overlap": len(tset & sset) / k,
        }
    return out


# ----------------------------------------------------------------------------
# H1c / H1d: parallel-lowrank (shared state, accept on REAL loss) vs sequential.
# Reuses a compact discrete coordinate descent on the GN surrogate for proposals.
# ----------------------------------------------------------------------------
def _coord_descent(lin, D_diag, F, resid, budget, sweeps=4, d0=None):
    m = lin.numel()
    d = torch.zeros(m, device=DEVICE, dtype=DTYPE) if d0 is None else d0.clone()
    a = resid + (F @ d if F.numel() else torch.zeros_like(resid))
    Fcol_sq = (F ** 2).sum(0) if F.numel() else torch.zeros(m, device=DEVICE, dtype=DTYPE)
    for _ in range(sweeps):
        changed = False
        for i in range(m):
            old = d[i].item(); nnz = int((d != 0).sum())
            best_v, best_g = old, 0.0
            for v in (-1.0, 0.0, 1.0):
                if v == old:
                    continue
                if old == 0.0 and v != 0.0 and nnz >= budget:
                    continue
                step = v - old
                dg = lin[i].item() * step + 0.5 * D_diag[i].item() * (v * v - old * old)
                if F.numel():
                    dg += (F[:, i] @ a).item() * step + 0.5 * Fcol_sq[i].item() * step * step
                if dg < best_g - 1e-12:
                    best_g, best_v = dg, v
            if best_v != old:
                step = best_v - old
                if F.numel():
                    a = a + F[:, i] * step
                d[i] = best_v; changed = True
        if not changed:
            break
    return d


def solve_parallel_lowrank_real(prob: L1Problem, c: Tensor, B: Tensor,
                                D_diag: Tensor, outer=6, trust=True):
    """Shared state u = B delta; per-layer local proposals; accept on REAL loss."""
    K, L = prob.cfg.budget_K, len(prob.layer_slices)
    lb = max(1, K // L)
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    cost = SolveCost()
    best = d.clone(); best_L = prob.loss_value(best); cost.real_forwards += 1
    for _ in range(outer):
        u = B @ d; cost.syncs += 1
        prop = d.clone()
        for sl in prob.layer_slices:
            Bl = B[:, sl]
            resid = u - Bl @ d[sl]
            prop[sl] = _coord_descent(c[sl], D_diag[sl], Bl, resid, lb, d0=d[sl])
        if int((prop != 0).sum()) > K:
            keep = torch.topk((c * prop).abs(), K).indices   # crude budget enforce
            m = torch.zeros_like(prop); m[keep] = prop[keep]; prop = m
        if trust:
            # accept only if surrogate improves; else shrink to best single coords
            if surrogate_dL(prob, c, B, prop) >= surrogate_dL(prob, c, B, d):
                changes = (prop != d).nonzero(as_tuple=True)[0]
                acc = d.clone(); cur = surrogate_dL(prob, c, B, acc)
                scored = sorted(
                    ((surrogate_dL(prob, c, B, _with(acc, i, prop[i])) - cur, i)
                     for i in changes.tolist()))
                for _, i in scored:
                    trial = _with(acc, i, prop[i])
                    if (surrogate_dL(prob, c, B, trial) < cur - 1e-12
                            and int((trial != 0).sum()) <= K):
                        acc, cur = trial, surrogate_dL(prob, c, B, trial)
                prop = acc
        Lp = prob.loss_value(prop); cost.real_forwards += 1
        if Lp < best_L - 1e-12:
            best_L, best, d = Lp, prop.clone(), prop.clone()
        else:
            lb = max(1, lb - 1); d = best.clone()
    return best, best_L, cost


def _with(d, i, v):
    t = d.clone(); t[i] = v; return t


def solve_sequential_real(prob: L1Problem):
    """Greedy on the REAL loss: each trial move is a real forward. Gold coord."""
    K = prob.cfg.budget_K
    d = torch.zeros(prob.M, device=DEVICE, dtype=DTYPE)
    cost = SolveCost()
    cur = prob.loss_value(d); cost.real_forwards += 1
    for _ in range(K):
        best_gain, best = 0.0, None
        for i in range(prob.M):
            if d[i] != 0:
                continue
            for v in (-1.0, 1.0):
                t = _with(d, i, v)
                Lt = prob.loss_value(t); cost.real_forwards += 1
                if Lt - cur < best_gain - 1e-12:
                    best_gain, best = Lt - cur, (i, v)
        if best is None:
            break
        d[best[0]] = best[1]; cur += best_gain; cost.syncs += 1
    return d, cur, cost


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------
def run_once(cfg: Level1Config) -> dict:
    prob = build_problem(cfg)
    c = candidate_gradient(prob)
    H_GN, R = exact_gn(prob)
    D_diag = torch.diagonal(H_GN).clone()
    ranks = list(cfg.sketch_ranks)

    audit = interaction_audit(prob, H_GN)
    h1b = h1b_real(prob, c, R, ranks)

    # Feasibility B: build a sketch at the top rank, count backwards.
    r_top = min(max(ranks), prob.M)
    B_sketch, sk_cost = sketch_gn(prob, r_top)
    # sketch fidelity vs exact GN
    sketch_relerr = float((B_sketch.T @ B_sketch - H_GN).norm() / H_GN.norm().clamp(min=1e-12))

    # H1c / H1d with REAL acceptance
    U, Svd, Vh = torch.linalg.svd(R, full_matrices=False)
    r_use = min(16, Svd.numel())
    B_oracle = Svd[:r_use].unsqueeze(1) * Vh[:r_use]
    d_par, L_par, c_par = solve_parallel_lowrank_real(prob, c, B_oracle, D_diag)
    d_seq, L_seq, c_seq = solve_sequential_real(prob)

    return {
        "cfg": cfg, "M": prob.M, "bits": cfg.bits,
        "interaction_audit": audit,
        "H1b_real": h1b,
        "feasibility_B": {
            "sketch_backwards": sk_cost.sketch_backwards,
            "sketch_vs_exactGN_relerr": sketch_relerr,
        },
        "H1c_H1d": {
            "parallel_lowrank_realL": L_par,
            "parallel_forwards": c_par.real_forwards,
            "sequential_realL": L_seq,
            "sequential_forwards": c_seq.real_forwards,
            "L_PTQ": audit["L_PTQ(baseline KL)"],
        },
    }


def _pretty(r: dict) -> None:
    print(f"\n==== M={r['M']} | bits={r['bits']} | device={DEVICE} ====")
    a = r["interaction_audit"]
    print(f"L_PTQ (teacher||student KL) = {a['L_PTQ(baseline KL)']:.6e}")
    print("\n[HEADLINE] GN vs TRUE interaction I_ij:")
    for name in ("cross_layer_pairs", "same_layer_pairs", "all_pairs"):
        d = a[name]
        if d.get("n", 0) < 3:
            print(f"   {name:<20}: n={d.get('n')}"); continue
        print(f"   {name:<20}: pearson={d['pearson_true_vs_GN']:+.3f}  "
              f"residual_relL2={d['residual_rel_L2']:.3f}  "
              f"sign_disagree={d['sign_disagree_rate']:.3f}")
    print("\n[H1b on REAL loss] (oracle low-rank of GN):")
    print(f"   {'rank':>5} | {'E(r)':>7} | {'spearman':>8} | {'topK':>6}")
    for rk, m in r["H1b_real"]["per_rank"].items():
        print(f"   {rk:>5} | {r['H1b_real']['E(r)'][rk]:>7.4f} | "
              f"{m['spearman']:>8.4f} | {m['topk_overlap']:>6.3f}")
    fb = r["feasibility_B"]
    print(f"\n[Feasibility B] sketch backwards={fb['sketch_backwards']}  "
          f"sketch_vs_exactGN_relerr={fb['sketch_vs_exactGN_relerr']:.4f}")
    h = r["H1c_H1d"]
    print("\n[H1c/H1d] real-loss after refinement (lower better):")
    print(f"   PTQ baseline        : {h['L_PTQ']:.6e}")
    print(f"   parallel_lowrank    : {h['parallel_lowrank_realL']:.6e}  "
          f"(forwards={h['parallel_forwards']})")
    print(f"   sequential_gold     : {h['sequential_realL']:.6e}  "
          f"(forwards={h['sequential_forwards']})")


if __name__ == "__main__":
    # Scenario W4: milder PTQ gap -> smaller residual -> GN should track I_ij well.
    _pretty(run_once(Level1Config(bits=4, seed=0)))

    # Scenario W3: coarser grid -> bigger residual. Watch residual_relL2 and
    # sign_disagree on CROSS-layer pairs climb -- that is GN's blind spot showing
    # up on its own, not injected. This is the test of concern #1.
    _pretty(run_once(Level1Config(bits=3, seed=0)))

    # Scenario W2: push it. If cross-layer residual explodes while E(r) stays
    # high, that is the clean falsification: compressible spectrum, wrong decisions.
    _pretty(run_once(Level1Config(bits=2, seed=0)))
