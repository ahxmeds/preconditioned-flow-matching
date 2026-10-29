"""Prepare web-sized figures for the project page (from the arXiv source)."""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "project-page", "assets", "figures")
os.makedirs(OUT, exist_ok=True)


def width_of(path):
    out = subprocess.run(["sips", "-g", "pixelWidth", path], capture_output=True, text=True).stdout
    return int(out.strip().split()[-1])


def sips(*args):
    r = subprocess.run(["sips", *args], capture_output=True, text=True)
    if r.returncode:
        print("sips failed:", " ".join(args), r.stderr[-300:])


def png(name, src, width=2000):
    dst = os.path.join(OUT, name)
    src_p = os.path.join(SRC, src)
    sips("-Z", str(min(width, width_of(src_p))), src_p, "--out", dst)
    print("png", name, os.path.getsize(dst) // 1024, "KB")


def jpg(name, src, width=1800, quality=84):
    dst = os.path.join(OUT, name)
    tmp = os.path.join(OUT, "_tmp.png")
    src_p = os.path.join(SRC, src)
    sips("-Z", str(min(width, width_of(src_p))), src_p, "--out", tmp)
    sips("-s", "format", "jpeg", "-s", "formatOptions", str(quality), tmp, "--out", dst)
    os.remove(tmp)
    print("jpg", name, os.path.getsize(dst) // 1024, "KB")


def render_pdf(name, src, width=2000):
    import fitz
    doc = fitz.open(os.path.join(SRC, src))
    page = doc[0]
    zoom = width / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    dst = os.path.join(OUT, name)
    pix.save(dst)
    print("pdf", name, pix.width, "x", pix.height, os.path.getsize(dst) // 1024, "KB")


def stack(name, top, bottom, labels=("no preconditioner", "FM preconditioner"), width=1500):
    a = mpimg.imread(os.path.join(SRC, top))
    b = mpimg.imread(os.path.join(SRC, bottom))
    h, w = a.shape[:2]
    fig, axes = plt.subplots(2, 1, figsize=(w / 150, 2 * h / 150 + 0.55), dpi=150)
    for ax, img, lab in zip(axes, [a, b], labels):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=13, loc="left", pad=3, color="#333333")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01, hspace=0.06)
    dst = os.path.join(OUT, name)
    fig.savefig(dst, facecolor="white", dpi=150)
    plt.close(fig)
    print("stack", name, os.path.getsize(dst) // 1024, "KB")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    render_pdf("fig1_concept.png", "Figures/fig1.pdf", 2200)
    png("fig2_gaussian.png", "Figures/gaussian_to_gaussian_without_and_with_preconditioning_with_labels.png", 2200)
    png("fig3_gmm.png", "Figures/row3_config2_standalone.png", 2400)
    png("fig5_kappa.png", "Figures/kappa_vs_t_compare.png", 1800)
    png("fig6_mnist.png", "Figures/comparison_panel_3methods.png", 2000)
    png("fig_ablation.png", "Figures/fid_vs_preconditioner_params.png", 1200)
    jpg("qual_churches.jpg", "Figures/churches_flow1_vs_flow12.png", 2000, 86)
    jpg("qual_flowers.jpg", "Figures/flowers_flow1_vs_flow12.png", 2000, 86)
    jpg("qual_cats.jpg", "Figures/afhq_cats_comparison_strip_specific.png", 1800, 85)
    jpg("qual_cifar.jpg", "Figures/cifar10_comparison.png", 1400, 88)
    stack("qual_imagenet.jpg", "Figures/imagenet64_baseline_1.png",
          "Figures/imagenet64_precond_1.png")
    print("total", sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) // 1024, "KB")
