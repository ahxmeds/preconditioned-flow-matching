#!/usr/bin/env python
"""Render the project-page concept video (2D Gaussian + Gaussian mixture).

Everything on screen comes from real runs of the released `pfm` code
(build/make_assets.py): learned ODE trajectories, generated samples, MMD curves.
The analytic parts (Sigma_t ellipses, kappa(t)) follow the paper's schedules.

    python build/make_video.py --out build/pfm_concept.mp4 --preview 3,14,33
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.collections import LineCollection                  # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402

import imageio_ffmpeg                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

W, H, FPS = 1920, 1080, 30

# ------------------------------------------------------------------ theme ----
BG     = "#0A0E16"
PANEL  = "#111826"
PANEL2 = "#0F1622"
EDGE   = "#1F2B3D"
TXT    = "#E8EEF6"
MUTED  = "#8695AB"
FAINT  = "#4E5D73"
BLUE   = "#4D9BFF"   # source / noise
GREEN  = "#2FD07B"   # data / target
AMBER  = "#FFB020"   # preconditioned target
PINK   = "#FF7BAC"   # model output
VIOLET = "#9D8CFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "text.color": TXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": EDGE,
    "savefig.facecolor": BG,
})

TIME_CMAP = LinearSegmentedColormap.from_list("time", [BLUE, "#7AA9E8", GREEN])


# ---------------------------------------------------------------- helpers ----
def new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def txt(fig, x, y, s, size=20, color=TXT, ha="left", va="baseline", weight="normal",
        alpha=1.0, z=5, **kw):
    return fig.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight,
                    alpha=alpha, zorder=z, **kw)


def header(fig, act):
    txt(fig, 0.042, 0.948, "PRECONDITIONED  FLOW  MATCHING", 15, FAINT)
    txt(fig, 0.958, 0.948, act, 15, FAINT, ha="right")
    fig.add_artist(Line2D([0.042, 0.958], [0.915, 0.915], color=EDGE, lw=1.1))


def caption(fig, line, sub=None, alpha=1.0):
    txt(fig, 0.042, 0.058, line, 25, TXT, alpha=alpha)
    if sub:
        txt(fig, 0.042, 0.022, sub, 18, MUTED, alpha=alpha)


def stage(fig, rect, lims):
    ax = fig.add_axes(rect)
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", adjustable="box")
    return ax


def panel(fig, rect, title="", xlabel="", ylabel="", logy=False):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL2)
    for sp in ax.spines.values():
        sp.set_color(EDGE)
        sp.set_linewidth(1.1)
    ax.tick_params(colors=MUTED, labelsize=14, length=3.5, width=1.0, pad=5)
    ax.grid(True, color=EDGE, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if logy:
        ax.set_yscale("log")
    if title:
        ax.set_title(title, color=TXT, fontsize=18, loc="left", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15, labelpad=6)
    return ax


def card(fig, x, y, w, h, accent=BLUE, alpha=1.0, fill=PANEL2):
    fig.add_artist(FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.003,rounding_size=0.010",
                                  transform=fig.transFigure, facecolor=fill,
                                  edgecolor=EDGE, lw=1.2, alpha=alpha, zorder=1))
    fig.add_artist(Line2D([x + 0.014, x + w - 0.014], [y + h - 0.0012] * 2, color=accent,
                          lw=3.0, alpha=alpha, zorder=2, solid_capstyle="round"))


def ellipse(ax, mean, cov, nsig=2.0, color=FAINT, lw=1.4, ls="--", alpha=0.9, z=4):
    evals, evecs = np.linalg.eigh(cov)
    ang = math.degrees(math.atan2(evecs[1, -1], evecs[0, -1]))
    w, h = 2 * nsig * np.sqrt(np.maximum(evals[::-1], 1e-12))
    e = Ellipse(tuple(mean), w, h, angle=ang, fill=False, edgecolor=color, lw=lw,
                ls=ls, alpha=alpha, zorder=z)
    ax.add_patch(e)
    return e


def trails(ax, traj, n=110, color="#6E7C93", alpha=0.30, lw=0.9, upto=None, z=3):
    idx = np.linspace(0, traj.shape[0] - 1, min(n, traj.shape[0])).astype(int)
    k = traj.shape[1] if upto is None else max(2, int(upto))
    segs = [traj[i, :k] for i in idx]
    ax.add_collection(LineCollection(segs, colors=color, alpha=alpha, linewidths=lw,
                                     zorder=z, capstyle="round"))


def dot_legend(ax, items, y=0.965, x=0.012, size=13, gap=0.20):
    """Direct legend inside a stage, in axes fraction coordinates."""
    for i, (label, color, marker) in enumerate(items):
        xx = x + i * gap
        ax.plot([xx], [y], transform=ax.transAxes, marker=marker, ms=9,
                mfc="none", mec=color, mew=1.8, ls="none", clip_on=False, zorder=9)
        ax.text(xx + 0.022, y, label, transform=ax.transAxes, fontsize=size,
                color=MUTED, va="center", ha="left", zorder=9)


def cov_of(x):
    xc = x - x.mean(0)
    return xc.T @ xc / max(len(x) - 1, 1)


def kappa_of(cov):
    ev = np.linalg.eigvalsh(cov)
    return float(ev[-1] / max(ev[0], 1e-12))


def lerp(a, b, w):
    return a + (b - a) * w


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


_DUR = 6.0


def fade_edges(fig, u, fin=0.22, fout=0.22):
    """Dip the frame out of / into the background at scene edges (absolute seconds)."""
    fin = min(fin / _DUR, 0.4)
    fout = min(fout / _DUR, 0.4)
    a = max(1.0 - u / fin, (u - (1 - fout)) / fout, 0.0)
    if a > 0:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor=BG, edgecolor="none", alpha=a, zorder=50))


# --------------------------------------------------------------- data load ---
def load(path):
    d = dict(np.load(path))
    return d


# ----------------------------------------------------------------- scenes ----
def scene_title(fig, u, D):
    header(fig, "NEURIPS 2026")
    a = ease(min(u / 0.35, 1.0))

    txt(fig, 0.5, 0.800, "Preconditioned", 66, TXT, ha="center", weight="bold", alpha=a)
    txt(fig, 0.5, 0.700, "Flow Matching", 66, "#BFD3EE", ha="center", weight="bold", alpha=a)
    txt(fig, 0.5, 0.612, "Reshaping the geometry that flow matching must regress",
        25, MUTED, ha="center", alpha=a)

    cards = [
        ("1  PRECONDITION", r"$x_1 \;\longrightarrow\; \tilde{x}_1$",
         "learn a reversible map  P", BLUE),
        ("2  MATCH", r"$\mathcal{N}(0,I) \;\longrightarrow\; \tilde{x}_1$",
         "train the flow in the new space", AMBER),
        ("3  MAP BACK", r"$\hat{x}_1 = P^{-1}(\hat{\tilde{x}}_1)$",
         "return to the data space", GREEN),
    ]
    w, h, y = 0.268, 0.148, 0.330
    for i, (lab, eq, sub, acc) in enumerate(cards):
        x = 0.083 + i * 0.288
        b = ease(min(max((u - 0.12 - 0.07 * i) / 0.30, 0.0), 1.0))
        card(fig, x, y, w, h, accent=acc, alpha=b)
        txt(fig, x + 0.022, y + h - 0.042, lab, 14, acc, alpha=b)
        txt(fig, x + 0.022, y + 0.062, eq, 25, TXT, alpha=b)
        txt(fig, x + 0.022, y + 0.028, sub, 14, MUTED, alpha=b)

    txt(fig, 0.5, 0.200, "NeurIPS 2026   ·   arXiv 2603.02337", 17, FAINT, ha="center")
    fade_edges(fig, u)


def scene_a1_intro(fig, u, D):
    header(fig, "01 / 04   ·   THE PROBLEM")
    lims = (-14, 66)
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)
    a = ease(min(u / 0.45, 1.0))

    x1 = D["x1"]
    x0 = D["x0"]
    ax.scatter(x1[:, 0], x1[:, 1], s=7, color=GREEN, alpha=0.55 * a, lw=0, zorder=3)
    ellipse(ax, D["mu"], D["cov"], 2.0, GREEN, 1.3, "--", 0.55 * a)
    ax.scatter(x0[:, 0], x0[:, 1], s=9, color=BLUE, alpha=0.55 * a, lw=0, zorder=3)

    ax.annotate("noise\n" + r"$\mathcal{N}(0,I)$", xy=(0, 0), xytext=(0.10, 0.13),
                textcoords="axes fraction", color=BLUE, fontsize=17, ha="left",
                alpha=a, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0, alpha=0.6 * a))
    ax.annotate("target\n" + r"$\mathcal{N}(\mu,H)$,  $\kappa(H)=100$", xy=(44, 44),
                xytext=(0.62, 0.90), textcoords="axes fraction", color=GREEN,
                fontsize=17, ha="left", va="top", alpha=a,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0, alpha=0.6 * a))

    ax2 = panel(fig, (0.505, 0.150, 0.452, 0.720), title="")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)
    lines = [
        ("same MLP, same budget", TXT, 19, 0.90),
        ("", TXT, 12, 0.84),
        ("Flow matching regresses", MUTED, 21, 0.78),
        (r"$v_\theta(x_t,t)\;\approx\;x_1-x_0$", TXT, 24, 0.665),
        ("", TXT, 12, 0.60),
        ("along a path whose intermediate", MUTED, 21, 0.53),
        ("distributions carry the target's", MUTED, 21, 0.45),
        ("anisotropy.", MUTED, 21, 0.37),
        ("", TXT, 12, 0.31),
        (r"$\Sigma_t=\cos^2\!(\frac{\pi t}{2})\,I+\sin^2\!(\frac{\pi t}{2})\,H$",
         AMBER, 21, 0.22),
        (r"$\kappa(\Sigma_t)$:  1  $\rightarrow$  100", AMBER, 21, 0.12),
    ]
    for s, c, size, yy in lines:
        if s:
            ax2.text(0.06, yy, s, transform=ax2.transAxes, fontsize=size, color=c,
                     va="center", ha="left", alpha=a)
    caption(fig, "Standard flow matching must transport noise onto a strongly anisotropic target.",
            "Every intermediate distribution it regresses on inherits that anisotropy.", a)
    fade_edges(fig, u)


def scene_a1_path(fig, u, D):
    header(fig, "01 / 04   ·   THE PATH IS ILL-CONDITIONED")
    lims = (-14, 66)
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)

    s = ease(u)
    t = s
    c, sn = math.cos(0.5 * math.pi * t), math.sin(0.5 * math.pi * t)

    x0, x1 = D["x0"], D["x1"]
    xt = c * x0 + sn * x1
    cols = TIME_CMAP(float(np.clip(sn ** 2, 0, 1)))
    ax.scatter(xt[:, 0], xt[:, 1], s=7, color=cols, alpha=0.75, lw=0, zorder=3)

    cov = (c ** 2) * np.eye(2) + (sn ** 2) * D["cov"]
    mean = sn * D["mu"]
    ellipse(ax, mean, cov, 2.0, "#9FB2CC", 1.6, "--", 0.9)
    k = kappa_of(cov)
    ax.text(0.985, 0.035, f"$\\kappa(\\Sigma_t)$ = {k:5.1f}", transform=ax.transAxes,
            fontsize=20, color=AMBER, ha="right", va="bottom", zorder=9)
    ax.text(0.015, 0.035, f"t = {t:4.2f}", transform=ax.transAxes, fontsize=22,
            color=MUTED, ha="left", va="bottom", zorder=9)

    # right: kappa(t)
    axk = panel(fig, (0.505, 0.150, 0.452, 0.720), title=r"path conditioning   $\kappa(\Sigma_t)$",
                xlabel="time  t", ylabel=r"$\kappa$", logy=True)
    ts, kap = D["ts_kappa"], D["kappa_raw"]
    axk.plot(ts, kap, color=AMBER, lw=2.8, zorder=3)
    axk.axhline(1.0, color=FAINT, lw=1.0, ls=":")
    axk.set_xlim(0, 1)
    axk.set_ylim(0.55, 220)
    axk.plot([t], [max(k, 1.0)], "o", ms=11, mfc="none", mec="#FFFFFF", mew=2.0, zorder=6)
    axk.annotate("κ = 100", xy=(1.0, 100), xytext=(0.62, 155), color=AMBER, fontsize=17,
                 arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.0, alpha=0.7))

    caption(fig, "As t → 1 the regression problem becomes 100× stiffer in the narrow direction.",
            "Gradient steps fit the wide direction first; the narrow one keeps lagging.", 1.0)
    fade_edges(fig, u)


def scene_a1_result(fig, u, D):
    header(fig, "01 / 04   ·   WHAT GOES WRONG")
    lims = (-14, 66)
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)
    a = ease(min(u / 0.30, 1.0))

    # learned trajectories, drawn progressively
    tr = D["traj_raw"]
    k = int(np.clip(u * 1.15, 0, 1) * (tr.shape[1] - 1))
    trails(ax, tr, n=120, upto=k + 2)
    ax.scatter(D["x1"][:, 0], D["x1"][:, 1], s=6, color=GREEN, alpha=0.35, lw=0, zorder=3)
    ax.scatter(tr[:, k, 0], tr[:, k, 1], s=13, facecolors="none", edgecolors=PINK,
               lw=0.9, alpha=0.85 * a, zorder=5)
    ellipse(ax, D["mu"], D["cov"], 2.0, GREEN, 1.2, "--", 0.45)

    # right top: MMD during training
    axm = panel(fig, (0.505, 0.545, 0.452, 0.325), title="MMD during training",
                xlabel="step", ylabel="MMD", logy=True)
    mmd = D["mmd_gauss_raw"]
    steps = np.arange(mmd.shape[1]) * int(D["mmd_steps"])
    m = mmd.mean(0)
    axm.plot(steps, m, color=PINK, lw=2.4)
    axm.fill_between(steps, m - mmd.std(0), m + mmd.std(0), color=PINK, alpha=0.18, lw=0)
    i = min(int(u * len(steps)), len(steps) - 1)
    axm.plot([steps[i]], [m[i]], "o", ms=9, mfc="none", mec="#FFFFFF", mew=1.8)
    axm.set_xlim(0, steps[-1])
    axm.text(0.97, 0.86, f"{m[-1]:.3f}", transform=axm.transAxes, color=PINK,
             fontsize=13, ha="right")

    # right bottom: narrow-direction spread
    axh = panel(fig, (0.505, 0.150, 0.452, 0.305),
                title="spread along the narrow direction", xlabel=r"$u_{\min}^\top(x-\mu)\;/\;\sigma_{\min}$")
    cov = D["cov"]
    evals, evecs = np.linalg.eigh(cov)
    uvec = evecs[:, 0]
    sd = math.sqrt(evals[0])
    zt = (D["x1"] - D["mu"]) @ uvec / sd
    zg = (D["gen_raw"] - D["mu"]) @ uvec / sd
    bins = np.linspace(-3.6, 3.6, 46)
    ht, _ = np.histogram(zt, bins, density=True)
    hg, _ = np.histogram(zg, bins, density=True)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    axh.plot(ctr, ht, color=GREEN, lw=2.6, label="target")
    axh.plot(ctr, hg, color=PINK, lw=2.6, label="FM output")
    axh.fill_between(ctr, 0, hg, color=PINK, alpha=0.14, lw=0)
    axh.legend(fontsize=12, frameon=False, loc="upper right", labelcolor=TXT)
    axh.set_xlim(-3.6, 3.6)
    axh.text(0.03, 0.80, f"variance ratio  {np.var(zg)/np.var(zt):.2f}",
             transform=axh.transAxes, fontsize=13, color=MUTED)

    caption(fig, "The model reaches the wide direction but under-fills the narrow one.",
            "A small FM loss still hides a large velocity error where $\\sigma_i(t)$ is small.", a)
    fade_edges(fig, u)


def _morph(fig, u, D, src="gauss"):
    if src == "gauss":
        A, b = D["P_a"], D["P_b"]
        x1, mu, cov = D["x1"], D["mu"], D["cov"]
        lims0, lims1 = (-14, 66), (-5.2, 5.2)
    else:
        A, b = D["gmm_P_a"], D["gmm_P_b"]
        x1 = D["gmm_x1"]
        mu, cov = None, None
        lims0, lims1 = (-4, 22), None
    return A, b, x1, mu, cov, lims0, lims1


def scene_a2_intro(fig, u, D):
    header(fig, "02 / 04   ·   PRECONDITION")
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), (-14, 66))
    a = ease(min(u / 0.4, 1.0))
    ax.scatter(D["x1"][:, 0], D["x1"][:, 1], s=7, color=GREEN, alpha=0.5 * a, lw=0, zorder=3)
    ellipse(ax, D["mu"], D["cov"], 2.0, GREEN, 1.3, "--", 0.5 * a)
    ax.text(0.5, 0.5, r"$P$", transform=ax.transAxes, fontsize=110, color=AMBER,
            ha="center", va="center", alpha=0.16 * a, weight="bold")

    axc = panel(fig, (0.505, 0.150, 0.452, 0.720))
    axc.set_xticks([])
    axc.set_yticks([])
    axc.grid(False)
    rows = [
        ("precondition, then match", TXT, 21, 0.90),
        ("", TXT, 12, 0.84),
        ("Learn a reversible map  P", MUTED, 20, 0.78),
        ("that whitens the target:", MUTED, 20, 0.71),
        (r"$\tilde{x}_1 = P\,x_1,\;\; \mathrm{Cov}(\tilde{x}_1) = I$", TXT, 23, 0.60),
        ("", TXT, 12, 0.53),
        ("Same transport problem,", MUTED, 20, 0.44),
        ("expressed in a representation", MUTED, 20, 0.37),
        ("with unit conditioning.", MUTED, 20, 0.30),
        ("", TXT, 12, 0.24),
        (r"$\tilde{\Sigma}_t = (1-t)^2 I + t^2\,P H P^{\top}$", AMBER, 20, 0.15),
    ]
    for s_, c_, size, yy in rows:
        if s_:
            axc.text(0.06, yy, s_, transform=axc.transAxes, fontsize=size, color=c_,
                     va="center", ha="left", alpha=a)
    caption(fig, "Preconditioning does not change the model, only the space it learns in.",
            "For a Gaussian target,  P = H^{-1/2}  is exact whitening.", a)
    fade_edges(fig, u)


def scene_a2_morph(fig, u, D):
    header(fig, "02 / 04   ·   WHITENED TARGET")
    A, b = D["P_a"], D["P_b"]
    x1 = D["x1"]
    w = ease(min(u / 0.92, 1.0))
    M = (1 - w) * np.eye(2) + w * A
    bb = w * b
    z = x1 @ M.T + bb
    lims = (lerp(-14, -5.2, w), lerp(66, 5.2, w))
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)
    col = (np.array([0.18, 0.82, 0.48]) * (1 - w) + np.array([1.0, 0.69, 0.13]) * w)
    ax.scatter(z[:, 0], z[:, 1], s=7, color=col, alpha=0.7, lw=0, zorder=4)
    ellipse(ax, z.mean(0), cov_of(z), 2.0, AMBER, 1.4, "--", 0.6)
    ax.text(0.015, 0.965, r"$P$  applied", transform=ax.transAxes, fontsize=20,
            color=AMBER, va="top", alpha=min(w * 1.6, 1.0))
    ax.text(0.015, 0.905, "the target becomes isotropic", transform=ax.transAxes,
            fontsize=17, color=MUTED, va="top", alpha=min(w * 1.6, 1.0))

    axk = panel(fig, (0.505, 0.150, 0.452, 0.720), title="path conditioning", logy=True,
                xlabel="time  t", ylabel=r"$\kappa$")
    ts, kap = D["ts_kappa"], D["kappa_raw"]
    axk.plot(ts, kap, color=FAINT, lw=2.0, ls="--")
    axk.plot(ts, np.ones_like(ts), color=AMBER, lw=3.0)
    axk.set_xlim(0, 1)
    axk.set_ylim(0.55, 220)
    axk.text(0.06, 0.30, r"$\kappa(\tilde{\Sigma}_t)\;\equiv\;1$", color=AMBER, fontsize=24,
             ha="left", transform=axk.transAxes)
    axk.annotate("before:  100", xy=(0.55, 100), xytext=(0.42, 0.62),
                 textcoords="axes fraction", color=FAINT, fontsize=14,
                 arrowprops=dict(arrowstyle="-", color=FAINT, lw=1.0))

    caption(fig, "The same data, viewed in a representation whose covariance is the identity.",
            "Every intermediate distribution of the preconditioned path stays perfectly conditioned.", 1.0)
    fade_edges(fig, u)


def scene_a2_transport(fig, u, D):
    header(fig, "02 / 04   ·   MATCH IN THE NEW SPACE")
    lims = (-5.2, 5.2)
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)
    tr = D["traj_pre"]
    k = int(np.clip(u * 1.1, 0, 1) * (tr.shape[1] - 1))
    trails(ax, tr, n=110, upto=k + 2, color="#7E8CA0", alpha=0.25)
    ax.scatter(D["x1_pre"][:, 0], D["x1_pre"][:, 1], s=10, facecolors="none",
               edgecolors=AMBER, lw=0.9, alpha=0.55, zorder=3)
    ax.scatter(tr[:, k, 0], tr[:, k, 1], s=9, color=PINK, alpha=0.75, lw=0, zorder=5)
    ax.text(0.015, 0.965, "noise  →  preconditioned target", transform=ax.transAxes,
            fontsize=19, color=MUTED, va="top")

    axk = panel(fig, (0.505, 0.545, 0.452, 0.325), title="conditioning along the path",
                logy=True, xlabel="time  t", ylabel=r"$\kappa$")
    ts, kap = D["ts_kappa"], D["kappa_raw"]
    axk.plot(ts, kap, color=FAINT, lw=1.8, ls="--")
    axk.plot(ts, np.ones_like(ts), color=AMBER, lw=3.0)
    axk.set_xlim(0, 1)
    axk.set_ylim(0.55, 220)
    axk.text(0.06, 0.16, "preconditioned", color=AMBER, fontsize=13, transform=axk.transAxes)
    axk.text(0.06, 0.80, "standard FM", color=FAINT, fontsize=13, transform=axk.transAxes)

    axm = panel(fig, (0.505, 0.150, 0.452, 0.305), title="MMD during training (3 runs)",
                logy=True, xlabel="step", ylabel="MMD")
    steps = np.arange(D["mmd_gauss_pre"].shape[1]) * int(D["mmd_steps"])
    for key, color, lab in [("mmd_gauss_raw", PINK, "standard FM"),
                            ("mmd_gauss_pre", AMBER, "preconditioned")]:
        v = D[key]
        m, se = v.mean(0), v.std(0) / math.sqrt(max(len(v), 1))
        axm.plot(steps, m, color=color, lw=2.4, label=lab)
        axm.fill_between(steps, np.maximum(m - se, 1e-6), m + se, color=color, alpha=0.16, lw=0)
    axm.legend(fontsize=12, frameon=False, loc="upper right", labelcolor=TXT)
    axm.set_xlim(0, steps[-1])

    caption(fig, "Trained in the whitened space, the same architecture converges faster and lower.",
            "Nothing about the model changed — only the geometry of its regression problem.", 1.0)
    fade_edges(fig, u)


def scene_a2_back(fig, u, D):
    header(fig, "02 / 04   ·   MAP BACK")
    w = ease(min(u / 0.85, 1.0))
    Ainv = np.linalg.inv(D["P_a"])
    # decode(z) = A^{-1}(z - b); animate from z-space to data space while the
    # camera follows, so the viewer sees P^{-1} act on the sample cloud.
    zt = D["x1_pre"]
    # interpolate: from zt (in z space) to P^{-1}(zt) (in data space), camera follows
    z_end = (zt - D["P_b"]) @ Ainv.T
    x = (1 - w) * zt + w * z_end
    lims = (lerp(-5.2, -14, w), lerp(5.2, 66, w))
    ax = stage(fig, (0.042, 0.150, 0.405, 0.720), lims)
    ax.scatter(D["x1"][:, 0], D["x1"][:, 1], s=7, color=GREEN, alpha=0.45, lw=0, zorder=3)
    ellipse(ax, D["mu"], D["cov"], 2.0, GREEN, 1.3, "--", 0.45)
    ax.scatter(x[:, 0], x[:, 1], s=11, facecolors="none", edgecolors=PINK, lw=0.9,
               alpha=0.8, zorder=5)
    ax.text(0.015, 0.965, r"$\hat{x}_1 = P^{-1}(\hat{\tilde{x}}_1)$", transform=ax.transAxes,
            fontsize=23, color=PINK, va="top")

    axm = panel(fig, (0.505, 0.150, 0.452, 0.720), title="final MMD (lower is better)")
    axm.set_xticks([])
    vals = [np.mean(D["mmd_gauss_raw"][:, -1]), np.mean(D["mmd_gauss_pre"][:, -1])]
    labs = ["standard FM", "preconditioned"]
    cols = [PINK, AMBER]
    for i, (v, lab, c) in enumerate(zip(vals, labs, cols)):
        yy = 0.72 - i * 0.30
        axm.barh([yy], [v], height=0.13, color=c, alpha=0.9, zorder=3)
        axm.text(0.03, yy + 0.085, lab, fontsize=16, color=MUTED, transform=axm.transAxes)
        axm.text(v + 0.00012, yy, f"{v:.4f}", fontsize=17, color=TXT, va="center",
                 ha="left")
    vmax = max(vals) * 1.35
    axm.set_xlim(0, vmax)
    axm.set_ylim(0, 1)
    axm.set_yticks([])
    axm.set_xticks(np.linspace(0, max(vals) * 1.2, 4).round(4))
    axm.tick_params(labelsize=11)
    axm.grid(True, axis="x", color=EDGE, lw=0.7)

    caption(fig, "The generated cloud lands on the original target, not on the whitened one.",
            "Preconditioning is transparent: the data distribution is unchanged.", 1.0)
    fade_edges(fig, u)


def scene_gmm(fig, u, D):
    header(fig, "03 / 04   ·   MIXTURES")
    lims = (-4, 22)
    axl = stage(fig, (0.055, 0.190, 0.365, 0.650), lims)
    axr = stage(fig, (0.560, 0.190, 0.365, 0.650), lims)

    tr_l = D["gmm_traj_raw"]
    tr_r = D["gmm_traj_pre_back"]
    k = int(np.clip(u * 1.05, 0, 1) * (tr_l.shape[1] - 1))

    for ax, tr, title, note in [(axl, tr_l, "standard FM", r"trained on  $x_1$"),
                                (axr, tr_r, "preconditioned FM", r"mapped back with  $P^{-1}$")]:
        ax.scatter(D["gmm_x1"][:, 0], D["gmm_x1"][:, 1], s=5, color=GREEN,
                   alpha=0.30, lw=0, zorder=2)
        trails(ax, tr, n=100, upto=k + 2, color="#7E8CA0", alpha=0.22)
        ax.scatter(tr[:, k, 0], tr[:, k, 1], s=8, color=PINK, alpha=0.70, lw=0, zorder=5)
        ax.text(0.5, 1.045, title, transform=ax.transAxes, fontsize=24, color=TXT,
                ha="center", va="bottom")
        ax.text(0.5, 1.005, note, transform=ax.transAxes, fontsize=15, color=MUTED,
                ha="center", va="bottom")

    axl.text(0.02, 0.03, r"$\kappa_{\max}$ = %.0f" % D["gmm_kappa_max_raw"], transform=axl.transAxes,
             fontsize=17, color=PINK)
    axr.text(0.02, 0.03, r"$z$-space:  $\mathrm{Cov}(\tilde{x}_1) = I$", transform=axr.transAxes,
             fontsize=17, color=AMBER)

    caption(fig, "Three anisotropic components, same budget on both sides.",
            "The thin component is the bottleneck; the mixture does not average it away.", 1.0)
    fade_edges(fig, u)


def scene_gmm_result(fig, u, D):
    header(fig, "03 / 04   ·   MIXTURES")
    lims = (-4, 22)
    axl = stage(fig, (0.055, 0.175, 0.365, 0.640), lims)
    axr = stage(fig, (0.560, 0.175, 0.365, 0.640), lims)

    a = ease(min(u / 0.4, 1.0))
    for ax, key, title, mmd_key in [
            (axl, "gmm_gen_raw", "standard FM", "mmd_gmm_raw"),
            (axr, "gmm_gen_pre", "preconditioned FM", "mmd_gmm_pre")]:
        ax.scatter(D["gmm_x1"][:, 0], D["gmm_x1"][:, 1], s=5, color=GREEN, alpha=0.30, lw=0, zorder=2)
        g = D[key]
        ax.scatter(g[:, 0], g[:, 1], s=9, facecolors="none", edgecolors=PINK, lw=0.8,
                   alpha=0.8 * a, zorder=5)
        ax.text(0.5, 1.045, title, transform=ax.transAxes, fontsize=24, color=TXT,
                ha="center", va="bottom")
        mmd = float(np.mean(D[mmd_key][:, -1]))
        ax.text(0.5, 1.005, f"MMD  {mmd:.3f}", transform=ax.transAxes, fontsize=16,
                color=AMBER if "pre" in key else PINK, ha="center", va="bottom")

    # measured spread of the thin modes (nearest-mean assignment, per component)
    means, covs = D["gmm_means"], D["gmm_covs"]

    def thin_spread(x):
        a_ = np.linalg.norm(x[:, None, :] - means[None], axis=2).argmin(1)
        out = []
        for k in range(len(means)):
            ev, evec = np.linalg.eigh(covs[k])
            u = evec[:, 0] / np.sqrt(ev[0])
            sel = a_ == k
            if sel.sum() > 20:
                out.append(float(((x[sel] - means[k]) @ u).var()))
        return out

    raw_spread = thin_spread(D["gmm_gen_raw"])
    pre_spread = thin_spread(D["gmm_gen_pre"])
    axl.text(0.5, 0.005, "thin-axis spread  " + " / ".join(f"{v:.1f}×" for v in raw_spread),
             transform=axl.transAxes, fontsize=13, color=PINK, ha="center", va="bottom")
    axr.text(0.5, 0.005, "thin-axis spread  " + " / ".join(f"{v:.1f}×" for v in pre_spread),
             transform=axr.transAxes, fontsize=13, color=AMBER, ha="center", va="bottom")

    caption(fig, "At equal budget the thin modes stay several times too wide without preconditioning.",
            "Final MMD differs by roughly 4x; the gap is largest early in training.", a)
    fade_edges(fig, u)


def scene_evidence(fig, u, D):
    header(fig, "04 / 04   ·   EVIDENCE")
    ax = panel(fig, (0.045, 0.200, 0.435, 0.660), title="MMD during training (GMM, 3 runs)",
               logy=True, xlabel="step", ylabel="MMD")
    steps = np.arange(D["mmd_gmm_pre"].shape[1]) * int(D["mmd_steps"])
    for key, color, lab in [("mmd_gmm_raw", PINK, "standard FM"),
                            ("mmd_gmm_pre", AMBER, "preconditioned")]:
        v = D[key]
        m, se = v.mean(0), v.std(0) / math.sqrt(max(len(v), 1))
        ax.plot(steps, m, color=color, lw=2.6, label=lab)
        ax.fill_between(steps, np.maximum(m - se, 1e-6), m + se, color=color, alpha=0.16, lw=0)
    ax.legend(fontsize=13, frameon=False, loc="upper right", labelcolor=TXT)
    ax.set_xlim(0, steps[-1])

    # FID across datasets (paper Table 2)
    axb = panel(fig, (0.535, 0.200, 0.420, 0.660), title="FID across image datasets",
                xlabel="FID  (lower is better)")
    data = [("LSUN Churches", 19.53, 14.47), ("Oxford Flowers", 23.82, 22.64),
            ("AFHQ Cats 512²", 7.11, 4.86), ("CIFAR-10", 4.15, 4.02),
            ("ImageNet-1k 64²", 8.66, 4.98)]
    ys = np.arange(len(data))[::-1]
    for (name, base, pre), y in zip(data, ys):
        axb.barh(y + 0.19, base, height=0.34, color="#3A4557", zorder=3)
        axb.barh(y - 0.19, pre, height=0.34, color=AMBER, zorder=3)
        axb.text(base + 0.35, y + 0.19, f"{base:.2f}", fontsize=12, color=MUTED, va="center")
        axb.text(pre + 0.35, y - 0.19, f"{pre:.2f}", fontsize=12, color=TXT, va="center")
        axb.text(-8.4, y, name, fontsize=13, color=TXT, va="center", ha="left")
    axb.set_yticks([])
    axb.set_ylim(-0.7, len(data) - 0.3)
    axb.set_xlim(-9.0, 27)
    axb.set_xticks([0, 5, 10, 15, 20, 25])
    axb.grid(True, axis="x", color=EDGE, lw=0.7)

    caption(fig, "Better conditioning, better samples: consistent FID gains from 32×32 to 512×512.",
            "Compute-matched baselines improve only marginally — the gain is geometry, not capacity.", 1.0)
    fade_edges(fig, u)


def scene_end(fig, u, D):
    header(fig, "NEURIPS 2026")
    a = ease(min(u / 0.4, 1.0))
    txt(fig, 0.5, 0.760, "Preconditioned Flow Matching", 60, TXT, ha="center",
        weight="bold", alpha=a)
    txt(fig, 0.5, 0.690, "Same architecture.  Same budget.  Better-conditioned geometry.",
        26, "#BFD3EE", ha="center", alpha=a)

    steps = [("precondition", r"$x_1 \to \tilde{x}_1$", BLUE),
             ("match", r"$\mathcal{N}(0,I) \to \tilde{x}_1$", AMBER),
             ("map back", r"$\hat{x}_1 = P^{-1}(\hat{\tilde{x}}_1)$", GREEN)]
    for i, (lab, eq, acc) in enumerate(steps):
        x = 0.20 + i * 0.215
        txt(fig, x, 0.560, lab, 15, acc, alpha=a)
        txt(fig, x, 0.505, eq, 23, TXT, alpha=a)
        if i < 2:
            txt(fig, x + 0.175, 0.505, "→", 24, FAINT, alpha=a)

    txt(fig, 0.5, 0.395, "paper   arxiv.org/abs/2603.02337", 18, MUTED, ha="center", alpha=a)
    txt(fig, 0.5, 0.350, "code    github.com/ahxmeds/preconditioned-flow-matching", 18,
        MUTED, ha="center", alpha=a)
    txt(fig, 0.5, 0.290, "Shadab Ahamed · Eshed Gal · Md Shahriar Rahim Siddiqui · "
        "Simon Ghyselincks · Moshe Eliasof · Eldad Haber", 15, FAINT, ha="center", alpha=a)
    fade_edges(fig, u, fin=0.15, fout=0.45)


SCENES = [
    (0.0, 5.0, scene_title),
    (5.0, 11.0, scene_a1_intro),
    (11.0, 20.0, scene_a1_path),
    (20.0, 26.5, scene_a1_result),
    (26.5, 29.5, scene_a2_intro),
    (29.5, 34.5, scene_a2_morph),
    (34.5, 41.0, scene_a2_transport),
    (41.0, 46.5, scene_a2_back),
    (46.5, 53.0, scene_gmm),
    (53.0, 57.5, scene_gmm_result),
    (57.5, 64.5, scene_evidence),
    (64.5, 72.0, scene_end),
]


def scene_at(t):
    global _DUR
    for t0, t1, fn in SCENES:
        if t0 <= t < t1:
            _DUR = t1 - t0
            return fn, (t - t0) / (t1 - t0)
    return SCENES[-1][2], 1.0


# --------------------------------------------------------------- auditing ----
def audit(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    items = []
    for t in fig.texts:
        try:
            items.append((t.get_text().strip(), t.get_window_extent(fig.canvas.get_renderer())))
        except Exception:
            pass
    for ax in fig.axes:
        for t in ax.texts:
            try:
                items.append((t.get_text().strip(), t.get_window_extent(fig.canvas.get_renderer())))
            except Exception:
                pass
        if ax.get_title():
            items.append((ax.get_title(), ax.title.get_window_extent(fig.canvas.get_renderer())))
    problems = []
    for ax in fig.axes:
        ab = ax.get_window_extent(fig.canvas.get_renderer())
        for t in list(ax.texts) + ([ax.title] if ax.get_title() else []):
            try:
                bb = t.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            above = t.get_transform() is ax.transAxes and t.get_position()[1] > 1.0
            if above:
                continue
            if bb.x0 < ab.x0 - 4 or bb.x1 > ab.x1 + 4 or bb.y0 < ab.y0 - 4 or bb.y1 > ab.y1 + 4:
                if t.get_text().strip():
                    problems.append(("outside-axes", t.get_text().strip()[:40]))
    for s, bb in items:
        if s and (bb.x0 < 2 or bb.y0 < 2 or bb.x1 > w - 2 or bb.y1 > h - 2):
            problems.append(("outside", s[:40]))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a.overlaps(b):
                ox = min(a.x1, b.x1) - max(a.x0, b.x0)
                oy = min(a.y1, b.y1) - max(a.y0, b.y0)
                if ox > 3 and oy > 3:
                    problems.append(("overlap", items[i][0][:32], items[j][0][:32]))
    return problems


def ascii_frame(path, cols=76):
    img = plt.imread(path)
    if img.shape[2] == 4:
        rgb = img[:, :, :3] * img[:, :, 3:4] + np.array([0.04, 0.055, 0.086]) * (1 - img[:, :, 3:4])
    else:
        rgb = img[:, :, :3]
    lum = rgb.mean(2)
    Hh, Ww = lum.shape
    rows = int(cols * Hh / Ww * 0.5)
    bs = Hh // rows
    bc = Ww // cols
    small = lum[:rows * bs, :cols * bc].reshape(rows, bs, cols, bc).mean(axis=(1, 3))
    ramp = " .:-=+*#%@"
    out = []
    for r in small:
        out.append("".join(ramp[min(int(v * (len(ramp) - 1) * 1.6), len(ramp) - 1)] for v in r))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=os.path.join(HERE, "assets2d.npz"))
    p.add_argument("--out", default=os.path.join(HERE, "pfm_concept.mp4"))
    p.add_argument("--preview", default="", help="comma-separated times (s) to dump as PNG")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--fps", type=int, default=FPS)
    args = p.parse_args()

    D = load(args.assets)
    meta_path = args.assets.replace(".npz", ".json")
    eval_every = 20
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            eval_every = int(json.load(f).get("gauss", {}).get("eval_every", 20))
    D["mmd_steps"] = eval_every
    D["gmm_kappa_max_raw"] = float(np.nanmax(D["gmm_kappa_raw"]))

    if args.preview:
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for ts in args.preview.split(","):
            t = float(ts)
            fn, u = scene_at(t)
            fig = new_fig()
            fn(fig, u, D)
            if args.audit:
                for pr in audit(fig):
                    print("AUDIT", t, pr)
            path = os.path.join(HERE, "preview", f"f{t:06.2f}.png")
            fig.savefig(path, dpi=100, facecolor=BG)
            plt.close(fig)
            print("wrote", path)
            print(ascii_frame(path))
        return

    total = int(SCENES[-1][1] * args.fps)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    problems = 0
    for i in range(total):
        t = i / args.fps
        fn, u = scene_at(t)
        fig = new_fig()
        fn(fig, u, D)
        fig.canvas.draw()
        if i % 12 == 0:
            pr = audit(fig)
            problems += len(pr)
            for x in pr[:3]:
                print(f"[audit {t:6.2f}s]", x)
        buf = np.asarray(fig.canvas.buffer_rgba())
        proc.stdin.write(buf.tobytes())
        plt.close(fig)
        if i % 150 == 0:
            print(f"  frame {i}/{total}  t={t:5.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("audit problems:", problems)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
