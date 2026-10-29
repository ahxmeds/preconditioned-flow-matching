"""Generate the real 2D results behind the project-page concept video.

Reuses the released `pfm` package (same targets, schedules, MLP and training
loop as `experiments/gaussian2d.py` and `experiments/gmm.py`) so the animation
shows the paper's own setting rather than a hand-drawn cartoon.

    python build/make_assets.py                  # both experiments
    python build/make_assets.py --only gmm       # one of them
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# in the repository the package sits at the root (../../pfm); in the author's
# scratch checkout it is mirrored under build/pfm_code.
for _cand in (os.path.abspath(os.path.join(HERE, "..", "..")),
              os.path.join(HERE, "..", "pfm_code")):
    if os.path.isdir(os.path.join(_cand, "pfm")):
        sys.path.insert(0, _cand)
        break

from pfm.flow import fm_loss, ode_sample                     # noqa: E402
from pfm.metrics import mmd_rbf                               # noqa: E402
from pfm.models import MLPVectorField                         # noqa: E402
from pfm.preconditioners import AffinePreconditioner          # noqa: E402
from pfm.schedules import SINCOS                              # noqa: E402
from pfm.utils import set_seed                                # noqa: E402


# ---------------------------------------------------------------- targets ----
class GMM:
    """Minimal mixture sampler (same recipe as experiments/gmm.py)."""

    def __init__(self, means, covs, weights):
        self.means, self.covs, self.weights = means, covs, weights
        self.k = len(means)

    def sample(self, n):
        idx = torch.multinomial(self.weights, n, replacement=True)
        chol = torch.linalg.cholesky(self.covs)
        eps = torch.randn(n, self.means.size(1))
        return self.means[idx] + torch.einsum("nij,nj->ni", chol[idx], eps)


def _cov(theta, hi, lo):
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]])
    return rot @ torch.diag(torch.tensor([hi, lo])) @ rot.t()


def gaussian_target():
    """Paper Fig. 2 setup: N((30,30), H), H = R diag(100, 1) R^T, 135 deg."""
    theta = 3 * math.pi / 4
    rot = torch.tensor([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]], dtype=torch.float32)
    cov = rot @ torch.diag(torch.tensor([100.0, 1.0])) @ rot.t()
    return torch.distributions.MultivariateNormal(torch.tensor([30.0, 30.0]), cov)


def gmm_target():
    """`three_comp` configuration of experiments/gmm.py (paper Fig. 3)."""
    means = torch.tensor([[7.5, 8.0], [16.2, 8.6], [11.5, 15.2]])
    covs = torch.stack([_cov(0.20, 2.4, 0.20), _cov(0.95, 3.8, 0.055), _cov(1.55, 1.9, 0.26)])
    return GMM(means, covs, torch.tensor([0.32, 0.36, 0.32]))


# --------------------------------------------------------------- training ----
def train(vf, target_sampler, steps, eval_every, eval_fn, batch=256, lr=1e-3):
    """Conditional flow matching on N(0, I) -> target_sampler(n)."""
    opt = torch.optim.Adam(vf.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        x0 = torch.randn(batch, 2)
        x1 = target_sampler(batch)
        t = torch.rand(batch)
        loss = fm_loss(vf, x0, x1, t, SINCOS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            hist.append(eval_fn(step, vf))
    vf.eval()
    return hist


def ode_traj(vf, x0, steps=120):
    """Euler integration recording the full particle path (n, steps+1, 2)."""
    x = x0.clone()
    out = [x.clone()]
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            ti = torch.full((x.size(0),), i * dt)
            x = x + dt * vf(x, ti)
            out.append(x.clone())
    return torch.stack(out, 1)


def decode_traj(precond, traj):
    """Apply the inverse preconditioner to every step of an (n, T, 2) trajectory."""
    n, T, d = traj.shape
    return precond.decode(traj.reshape(n * T, d)).reshape(n, T, d)


def kappa_curve_gaussian(num_t=101):
    ts = np.linspace(0, 1, num_t)
    c, s = np.cos(0.5 * np.pi * ts) ** 2, np.sin(0.5 * np.pi * ts) ** 2
    return ts, (c + s * 100.0) / (c + s * 1.0)


# ------------------------------------------------------------- experiments ---
def run_gauss(out, meta, args):
    target = gaussian_target()

    def san(n):
        return target.sample((n,))

    ts_k, kap_raw = kappa_curve_gaussian()
    out["ts_kappa"] = ts_k.astype(np.float32)
    out["kappa_raw"] = kap_raw.astype(np.float32)
    out["kappa_pre"] = np.ones_like(kap_raw, dtype=np.float32)

    precond = AffinePreconditioner.zca(san(50000))
    out["P_a"], out["P_b"] = precond.a.numpy(), precond.b.numpy()

    eval_x0 = torch.randn(args.n_eval, 2)
    eval_x1 = san(args.n_eval)
    out["x0"], out["x1"] = eval_x0.numpy(), eval_x1.numpy()
    out["x1_pre"] = precond.encode(eval_x1).numpy()
    out["mu"] = target.mean.numpy()
    out["cov"] = target.covariance_matrix.numpy()

    mmd_raw, mmd_pre = [], []
    for seed in range(args.seeds):
        set_seed(1000 + seed)
        set_seed(seed)

        def ev_raw(step, vf):
            g = ode_sample(vf, eval_x0, 100, "euler")
            return dict(step=step, mmd=float(mmd_rbf(g, eval_x1, sigma=2.0)), gen=g)

        vf_raw = MLPVectorField(dim=2)
        h_raw = train(vf_raw, san, args.steps, args.eval_every, ev_raw)
        mmd_raw.append([e["mmd"] for e in h_raw])
        if seed == 0:
            out["gen_raw"] = h_raw[-1]["gen"].numpy()
            out["traj_raw"] = ode_traj(vf_raw, torch.randn(args.n_anim, 2)).numpy()
        torch.save(vf_raw.state_dict(), os.path.join(HERE, f"vf_gauss_raw_{seed}.pt"))

        set_seed(2000 + seed)

        def ev_pre(step, vf):
            g = precond.decode(ode_sample(vf, eval_x0, 100, "euler"))
            return dict(step=step, mmd=float(mmd_rbf(g, eval_x1, sigma=2.0)), gen=g)

        vf_pre = MLPVectorField(dim=2)
        h_pre = train(vf_pre, lambda n: precond.encode(san(n)), args.steps,
                      args.eval_every, ev_pre)
        mmd_pre.append([e["mmd"] for e in h_pre])
        if seed == 0:
            out["gen_pre"] = h_pre[-1]["gen"].numpy()
            tr = ode_traj(vf_pre, torch.randn(args.n_anim, 2)).numpy()
            out["traj_pre"] = tr
            out["traj_pre_back"] = decode_traj(precond, torch.from_numpy(tr)).numpy()
        torch.save(vf_pre.state_dict(), os.path.join(HERE, f"vf_gauss_pre_{seed}.pt"))
        print(f"[gauss seed {seed}] mmd raw={h_raw[-1]['mmd']:.4f} pre={h_pre[-1]['mmd']:.4f}",
              flush=True)

    out["mmd_gauss_raw"] = np.array(mmd_raw, dtype=np.float32)
    out["mmd_gauss_pre"] = np.array(mmd_pre, dtype=np.float32)
    meta["gauss"] = dict(steps=args.steps, eval_every=args.eval_every,
                         mmd_raw=float(np.mean([m[-1] for m in mmd_raw])),
                         mmd_pre=float(np.mean([m[-1] for m in mmd_pre])),
                         kappa_t1_raw=100.0, kappa_t1_pre=1.0)


def run_gmm(out, meta, args):
    gmm = gmm_target()
    out["gmm_means"], out["gmm_covs"] = gmm.means.numpy(), gmm.covs.numpy()
    gmm_pre = AffinePreconditioner.zca(gmm.sample(30000))
    out["gmm_P_a"], out["gmm_P_b"] = gmm_pre.a.numpy(), gmm_pre.b.numpy()

    def comp_kappa(covs, a=None, num_t=61):
        covs = covs if a is None else a @ covs @ a.t()
        eye = torch.eye(2)
        rows = []
        for t in torch.linspace(0, 1, num_t):
            sig = (1 - t) ** 2 * eye + t ** 2 * covs
            ev = torch.linalg.eigvalsh(sig)
            rows.append((ev[:, -1] / ev[:, 0]).numpy())
        return np.stack(rows)

    out["gmm_kappa_raw"] = comp_kappa(gmm.covs).astype(np.float32)
    out["gmm_kappa_pre"] = comp_kappa(gmm.covs, gmm_pre.a).astype(np.float32)
    out["gmm_ts_kappa"] = np.linspace(0, 1, out["gmm_kappa_raw"].shape[0]).astype(np.float32)

    g_eval_x0 = torch.randn(args.n_eval, 2)
    g_eval_x1 = gmm.sample(args.n_eval)
    out["gmm_x0"], out["gmm_x1"] = g_eval_x0.numpy(), g_eval_x1.numpy()
    out["gmm_x1_pre"] = gmm_pre.encode(g_eval_x1).numpy()

    g_mmd_raw, g_mmd_pre = [], []
    for seed in range(args.seeds):
        set_seed(3000 + seed)

        def g_raw(step, vf):
            g = ode_sample(vf, g_eval_x0, 100, "rk4")
            return dict(step=step, mmd=float(mmd_rbf(g, g_eval_x1, sigma=[1.0, 2.0, 4.0])), gen=g)

        vf = MLPVectorField(dim=2)
        hg = train(vf, gmm.sample, args.gmm_steps, args.eval_every, g_raw)
        g_mmd_raw.append([e["mmd"] for e in hg])
        if seed == 0:
            out["gmm_gen_raw"] = hg[-1]["gen"].numpy()
            out["gmm_traj_raw"] = ode_traj(vf, torch.randn(args.n_anim, 2)).numpy()
        torch.save(vf.state_dict(), os.path.join(HERE, f"vf_gmm_raw_{seed}.pt"))

        set_seed(4000 + seed)

        def g_pre(step, vf):
            g = gmm_pre.decode(ode_sample(vf, g_eval_x0, 100, "rk4"))
            return dict(step=step, mmd=float(mmd_rbf(g, g_eval_x1, sigma=[1.0, 2.0, 4.0])), gen=g)

        vf2 = MLPVectorField(dim=2)
        hg2 = train(vf2, lambda n: gmm_pre.encode(gmm.sample(n)), args.gmm_steps,
                    args.eval_every, g_pre)
        g_mmd_pre.append([e["mmd"] for e in hg2])
        if seed == 0:
            out["gmm_gen_pre"] = hg2[-1]["gen"].numpy()
            tr = ode_traj(vf2, torch.randn(args.n_anim, 2)).numpy()
            out["gmm_traj_pre"] = tr
            out["gmm_traj_pre_back"] = decode_traj(gmm_pre, torch.from_numpy(tr)).numpy()
        torch.save(vf2.state_dict(), os.path.join(HERE, f"vf_gmm_pre_{seed}.pt"))
        print(f"[gmm seed {seed}] mmd raw={hg[-1]['mmd']:.4f} pre={hg2[-1]['mmd']:.4f}",
              flush=True)

    out["mmd_gmm_raw"] = np.array(g_mmd_raw, dtype=np.float32)
    out["mmd_gmm_pre"] = np.array(g_mmd_pre, dtype=np.float32)
    meta["gmm"] = dict(steps=args.gmm_steps, eval_every=args.eval_every,
                       mmd_raw=float(np.mean([m[-1] for m in g_mmd_raw])),
                       mmd_pre=float(np.mean([m[-1] for m in g_mmd_pre])),
                       kappa_max_raw=float(np.max(out["gmm_kappa_raw"])),
                       kappa_max_pre=float(np.max(out["gmm_kappa_pre"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--gmm-steps", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-anim", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--only", choices=["all", "gauss", "gmm"], default="all")
    p.add_argument("--merge", action="store_true", help="extend an existing npz")
    p.add_argument("--out", default=os.path.join(HERE, "assets2d.npz"))
    args = p.parse_args()

    out, meta = {}, {}
    if args.merge and os.path.exists(args.out):
        out = dict(np.load(args.out))
        with open(args.out.replace(".npz", ".json")) as f:
            meta = json.load(f)

    if args.only in ("all", "gauss"):
        run_gauss(out, meta, args)
    if args.only in ("all", "gmm"):
        run_gmm(out, meta, args)

    np.savez_compressed(args.out, **out)
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
