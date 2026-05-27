"""Generate publication-grade architecture / methodology / result figures for the README.

Run:  python assets/make_figures.py
Outputs SVG (embedded in README) + high-DPI PNG (preview) into assets/.
One coherent visual language is shared by every figure (architecture + result plots).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np
import csv
import os

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
ASSET = os.path.join(ROOT, "webapp", "presentation", "slide_assets")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",      # outline text -> identical rendering everywhere
    "figure.dpi": 100,
})

# ---- palette -------------------------------------------------------------
INK     = "#1f2937"
MUTE    = "#6b7280"
PAGE    = "#ffffff"
FANC    = "#8b97a4"
GRIDC   = "#e5e7eb"
INPUT   = "#334155";  INPUT_BG  = "#eef2f7"
BACKB   = "#3b4ab0";  BACKB_BG  = "#e7eaff"
DEPTH   = "#15803d";  DEPTH_BG  = "#e3f5e8"
FUSE    = "#7c3aed";  FUSE_BG   = "#efe6ff"
NECK    = "#0e8f8d";  NECK_BG   = "#e0f5f4"
HEAD    = "#c2410c";  HEAD_BG   = "#fdeadd"
CAL     = "#b91c1c";  CAL_BG    = "#fde2e2"
DARK    = "#111827"
# result-plot series colors
C_RGBD  = "#3b4ab0"   # RGB-D primary  (indigo)
C_RGB   = "#c2410c"   # RGB comparison (burnt orange)

SHADOW = [pe.withSimplePatchShadow(offset=(1.4, -1.4), alpha=0.16, shadow_rgbFace="#0b1020")]


# ============================ schematic helpers ===========================
def box(ax, cx, cy, w, h, title, sub=None, fc="#fff", ec=INK, tc=INK,
        title_size=12.5, sub_size=9.0, lw=1.6, round_size=0.10, shadow=True, bold=True):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f"round,pad=0.012,rounding_size={round_size}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)
    if sub:
        ax.text(cx, cy + h*0.17, title, ha="center", va="center", fontsize=title_size,
                color=tc, fontweight="bold" if bold else "normal", zorder=4)
        ax.text(cx, cy - h*0.25, sub, ha="center", va="center", fontsize=sub_size,
                color=MUTE, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size,
                color=tc, fontweight="bold" if bold else "normal", zorder=4)
    return dict(cx=cx, cy=cy, w=w, h=h, L=(cx-w/2, cy), R=(cx+w/2, cy),
                T=(cx, cy+h/2), B=(cx, cy-h/2))


def fmap_stack(ax, cx, cy, label, sub, n=4, base_h=2.2, base_w=0.42,
               fc=BACKB_BG, ec=BACKB, gap=0.34):
    total_w = (n-1)*gap + base_w
    x0 = cx - total_w/2
    for i in range(n):
        frac = 1 - 0.62*(i/(n-1))
        h = base_h*frac
        x = x0 + i*gap
        p = FancyBboxPatch((x, cy-h/2), base_w, h,
                           boxstyle="round,pad=0.004,rounding_size=0.05", linewidth=1.3,
                           edgecolor=ec, facecolor=fc if i < n-1 else ec, zorder=3+i*0.01)
        if i == 0:
            p.set_path_effects(SHADOW)
        ax.add_patch(p)
    # ellipsis tick so the slab stack reads unambiguously as a CNN
    ax.text(x0 + total_w - gap*1.5, cy, "···", ha="center", va="center",
            fontsize=11, color=ec, zorder=6)
    ax.text(cx, cy - base_h/2 - 0.46, label, ha="center", va="center",
            fontsize=12.5, color=INK, fontweight="bold", zorder=5)
    ax.text(cx, cy - base_h/2 - 0.86, sub, ha="center", va="center",
            fontsize=8.6, color=MUTE, zorder=5)
    return dict(L=(x0, cy), R=(x0+total_w, cy), cx=cx, cy=cy, w=total_w, h=base_h)


def arrow(ax, p1, p2, color=INK, lw=2.0, rad=0.0, style="-|>", ls="-", shrinkB=4):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                 connectionstyle=f"arc3,rad={rad}", linewidth=lw, color=color, zorder=2,
                 linestyle=ls, shrinkA=3, shrinkB=shrinkB, capstyle="round"))


def group(ax, x0, y0, x1, y1, label, color=HEAD):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                 boxstyle="round,pad=0.02,rounding_size=0.16", linewidth=1.4, edgecolor=color,
                 facecolor="none", linestyle=(0, (5, 3)), zorder=1, alpha=0.9))
    ax.text(x0+0.1, y1+0.22, label, ha="left", va="bottom", fontsize=9.8, color=color,
            fontweight="bold", zorder=5)


def photo_glyph(ax, cx, cy, w=1.5, h=0.95, ec=INPUT):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.05", lw=1.2, ec=ec, fc="#ffffff", zorder=4))
    ax.add_patch(Circle((cx-w*0.26, cy+h*0.16), 0.13, color="#f59e0b", zorder=5))
    ax.plot([cx-w*0.42, cx-w*0.05, cx+w*0.18, cx+w*0.42],
            [cy-h*0.22, cy+h*0.20, cy-h*0.05, cy+h*0.24], color=DEPTH, lw=1.5, zorder=5,
            solid_capstyle="round")


def new_ax(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_facecolor(PAGE)
    return fig, ax


def style_axes(ax, title=None):
    ax.set_facecolor(PAGE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9ca3af"); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=9.5, length=3)
    ax.grid(axis="y", color=GRIDC, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=13.5, color=INK, fontweight="bold", loc="left", pad=10)


def save(fig, name, equal=True):
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight", pad_inches=0.10,
                    facecolor=PAGE, **kw)
    plt.close(fig)
    print("wrote", name)


def read_preds(run):
    f = os.path.join(ASSET, f"{run}_predictions_with_errors.csv")
    t, p = [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["target_calories"])); p.append(float(row["predicted_calories"]))
    return np.array(t), np.array(p)


def read_bins(run):
    f = os.path.join(ASSET, f"{run}_error_by_calorie_bin.csv")
    names, vals = [], []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            names.append(row["bin"]); vals.append(float(row["mean"]))
    return names, np.array(vals)


RGBD_RUN = "primary"
RGB_RUN  = "outputs_food101_4passes"


# =============================== Fig 1 ====================================
def fig_multitask():
    fig, ax = new_ax(13.6, 7.0, (0, 27), (0, 14))
    ax.text(0.4, 13.2, "Multi-task nutrition network", fontsize=16.5, color=INK,
            fontweight="bold", ha="left")
    ax.text(0.4, 12.45, "shared by Exp 1 (per-gram)  ·  Exp 2 (direct)  ·  Exp 3 (RGB-D)",
            fontsize=10.5, color=MUTE, ha="left")
    yc = 7.2
    iw, ih = 3.5, 3.5
    ip = FancyBboxPatch((2.7-iw/2, yc-ih/2), iw, ih, boxstyle="round,pad=0.012,rounding_size=0.10",
                        lw=1.6, ec=INPUT, fc=INPUT_BG, zorder=3); ip.set_path_effects(SHADOW)
    ax.add_patch(ip)
    photo_glyph(ax, 2.7, yc+0.78)
    ax.text(2.7, yc-0.55, "Food image", ha="center", va="center", fontsize=12.5, color=INPUT,
            fontweight="bold", zorder=4)
    ax.text(2.7, yc-1.12, "RGB 3ch · RGB-D 4ch", ha="center", va="center", fontsize=8.0,
            color=MUTE, zorder=4)
    bb = fmap_stack(ax, 8.4, yc, "InceptionV3", "ImageNet-1K  ·  299² → 8×8×2048",
                    n=5, base_h=3.0, base_w=0.5, gap=0.55)
    pool = box(ax, 13.4, yc, 2.3, 1.7, "GAP", "→ 2048-d", fc=NECK_BG, ec=NECK, tc=NECK,
               title_size=12.5, sub_size=9)
    trunk = box(ax, 16.9, yc, 3.0, 2.0, "Shared FC", "2048 → 512", fc=BACKB_BG, ec=BACKB,
                tc=BACKB, title_size=12.5, sub_size=9)
    arrow(ax, (2.7+iw/2, yc), (bb["L"][0]-0.15, yc))
    arrow(ax, bb["R"], pool["L"]); arrow(ax, pool["R"], trunk["L"])
    heads = [("Calories", "kcal · kcal/g", HEAD, HEAD_BG, HEAD),
             ("Mass", "g", HEAD, HEAD_BG, HEAD),
             ("Fat", "g · g/g", HEAD, HEAD_BG, HEAD),
             ("Carb", "g · g/g", HEAD, HEAD_BG, HEAD),
             ("Protein", "g · g/g", HEAD, HEAD_BG, HEAD)]
    hx = 22.3
    ys = np.linspace(yc+3.9, yc-3.9, len(heads))
    group(ax, hx-1.55, ys[-1]-0.92, hx+1.55, ys[0]+0.92, "Per-task heads  ·  FC 512→1", color=HEAD)
    for (name, unit, ec, bg, tc), y in zip(heads, ys):
        hb = box(ax, hx, y, 2.5, 1.4, name, unit, fc=bg, ec=ec, tc=tc, title_size=11,
                 sub_size=8.2, round_size=0.12)
        r = 0.14*np.sign(y-yc) if abs(y-yc) > 0.2 else 0.0
        arrow(ax, trunk["R"], hb["L"], color=FANC, lw=2.1, rad=r)
    # neutral target chips (single visual system: color encodes task only, not experiment)
    ax.text(0.4, 1.85, "Targets", fontsize=11, color=INK, fontweight="bold", ha="left")
    chips = ["Exp 1 · portion-independent → per-gram (kcal/g, g/g)",
             "Exp 2 · direct → absolute (kcal, g)",
             "Exp 3 · + depth as 4th input channel"]
    cx = 0.4
    for txt in chips:
        w = 0.158*len(txt) + 0.9
        box(ax, cx+w/2, 0.95, w, 0.92, txt, fc="#ffffff", ec="#94a3b8", tc=INK,
            title_size=9.0, lw=1.3, round_size=0.20, shadow=False, bold=False)
        cx += w + 0.5
    save(fig, "fig_multitask")


# =============================== Fig 2 ====================================
def fig_exp4():
    fig, ax = new_ax(13.2, 5.2, (0, 26.5), (0, 10.5))
    ax.text(0.4, 9.7, "Exp 4 — volume-scalar mass pipeline", fontsize=15.5, color=INK,
            fontweight="bold", ha="left")
    ov = box(ax, 3.0, 7.2, 3.9, 2.0, "Overhead RGB", "+ volume scalar", fc=INPUT_BG, ec=INPUT,
             tc=INPUT, title_size=11, sub_size=9)
    mr = box(ax, 9.4, 7.2, 4.9, 2.0, "MassRegressor", "InceptionV3 ⊕ volume → FC", fc=BACKB_BG,
             ec=BACKB, tc=BACKB, title_size=12.5, sub_size=9)
    mass = box(ax, 15.2, 7.2, 2.6, 1.5, "Mass", "ĝ (grams)", fc=HEAD_BG, ec=HEAD, tc=HEAD,
               title_size=11.5, sub_size=9)
    arrow(ax, ov["R"], mr["L"]); arrow(ax, mr["R"], mass["L"])
    sf = box(ax, 3.0, 3.0, 3.9, 2.0, "Side-angle frames", "4 cameras", fc=INPUT_BG, ec=INPUT,
             tc=INPUT, title_size=10, sub_size=9)
    pg = box(ax, 9.4, 3.0, 5.4, 2.0, "Exp 1 per-gram model", "frozen checkpoint", fc=DEPTH_BG,
             ec=DEPTH, tc=DEPTH, title_size=11, sub_size=9)
    calg = box(ax, 15.2, 3.0, 2.6, 1.5, "cal / g", "kcal per gram", fc=HEAD_BG, ec=HEAD, tc=HEAD,
               title_size=11.5, sub_size=9)
    arrow(ax, sf["R"], pg["L"]); arrow(ax, pg["R"], calg["L"])
    mul = Circle((19.3, 5.1), 0.62, facecolor="#ffffff", edgecolor=CAL, lw=2.0, zorder=4)
    mul.set_path_effects(SHADOW); ax.add_patch(mul)
    ax.text(19.3, 5.12, "×", ha="center", va="center", fontsize=20, color=CAL,
            fontweight="bold", zorder=5)
    arrow(ax, mass["R"], (18.8, 5.45), color=HEAD, rad=-0.10, shrinkB=2)
    arrow(ax, calg["R"], (18.8, 4.75), color=HEAD, rad=0.10, shrinkB=2)
    out = box(ax, 23.4, 5.1, 3.0, 1.8, "Calories", "kcal", fc=CAL_BG, ec=CAL, tc=CAL,
              title_size=13, sub_size=9.5)
    arrow(ax, (19.92, 5.1), out["L"], color=CAL, lw=2.2)
    save(fig, "fig_exp4")


# =============================== Fig 3 ====================================
def fig_dpf():
    fig, ax = new_ax(13.5, 6.4, (0, 28), (0, 13.4))
    ax.text(0.4, 12.7, "DPF-Nutrition — depth-prediction & dual-stream fusion", fontsize=15.5,
            color=INK, fontweight="bold", ha="left")
    yc = 6.2
    rgb = box(ax, 2.8, yc, 3.9, 1.7, "Overhead RGB", "rgb.png", fc=INPUT_BG, ec=INPUT, tc=INPUT,
              title_size=10.5, sub_size=9)
    da = box(ax, 7.3, yc+3.1, 4.4, 1.7, "Depth Anything V2", "frozen → depth", fc=DEPTH_BG,
             ec=DEPTH, tc=DEPTH, title_size=10.5, sub_size=9)
    rgb_s = fmap_stack(ax, 11.6, yc-2.1, "RGB stream", "ResNet-101 · 7×7×2048", n=4, base_h=2.0,
                       base_w=0.42, gap=0.42, fc=BACKB_BG, ec=BACKB)
    dep_s = fmap_stack(ax, 11.6, yc+2.1, "Depth stream", "ResNet-101 · 1-ch stem", n=4,
                       base_h=2.0, base_w=0.42, gap=0.42, fc=DEPTH_BG, ec=DEPTH)
    # start branch arrows from box corners so they clear the centred title text
    arrow(ax, (rgb["cx"]+1.0, rgb["T"][1]), (da["L"][0]-0.1, da["cy"]), color=DEPTH, rad=0.18)
    arrow(ax, (rgb["cx"]+1.0, rgb["B"][1]), rgb_s["L"], color=BACKB, rad=-0.12)
    arrow(ax, (da["R"][0], da["cy"]), dep_s["L"], color=DEPTH, rad=-0.1)
    cab = box(ax, 17.2, yc, 3.4, 3.6, "CAB fusion", "channel + spatial\nattention · same level",
              fc=FUSE_BG, ec=FUSE, tc=FUSE, title_size=12.5, sub_size=7.8)
    arrow(ax, rgb_s["R"], (cab["cx"]-cab["w"]/2, yc-0.9), color=BACKB, rad=0.12)
    arrow(ax, dep_s["R"], (cab["cx"]-cab["w"]/2, yc+0.9), color=DEPTH, rad=-0.12)
    ms = box(ax, 21.7, yc, 3.5, 1.9, "Multi-scale", "add → GAP", fc=NECK_BG, ec=NECK, tc=NECK,
             title_size=10.5, sub_size=9)
    arrow(ax, cab["R"], ms["L"], color=FUSE)
    heads = ["Calories", "Mass", "Fat", "Carb", "Protein"]
    cols  = [HEAD, HEAD, HEAD, HEAD, HEAD]
    hx = 25.4
    ys = np.linspace(yc+3.7, yc-3.7, len(heads))
    group(ax, hx-1.35, ys[-1]-0.7, hx+1.35, ys[0]+0.7, "5 heads", color=HEAD)
    for name, col, y in zip(heads, cols, ys):
        hb = box(ax, hx, y, 2.2, 1.05, name, fc=(CAL_BG if col == CAL else HEAD_BG), ec=col,
                 tc=col, title_size=10.5, round_size=0.14)
        r = 0.14*np.sign(y-yc) if abs(y-yc) > 0.2 else 0.0
        arrow(ax, ms["R"], hb["L"], color=FANC, lw=2.0, rad=r)
    ax.text(0.4, 1.2, r"Geometric-mean L1 loss:   $\mathcal{L}=\sqrt[5]{\,L_{cal}\,L_{mass}\,L_{fat}\,L_{carb}\,L_{prot}\,}$",
            fontsize=11, color=INK, ha="left")
    save(fig, "fig_dpf")


# =============================== Fig 4 ====================================
def fig_pipeline():
    fig, ax = new_ax(13.6, 5.6, (0, 27.5), (0, 11.5))
    ax.text(0.4, 10.8, "Methodology — compute & data pipeline", fontsize=15.5, color=INK,
            fontweight="bold", ha="left")
    yc = 7.6
    local = box(ax, 2.7, yc, 3.1, 1.8, "Local Mac", "edit code · git", fc=INPUT_BG, ec=INPUT,
                tc=INPUT, title_size=12, sub_size=9)
    gh = box(ax, 7.9, yc, 3.6, 1.8, "GitHub", "T0MYYY / Nutrition5k", fc="#eef2f7", ec=DARK,
             tc=DARK, title_size=12, sub_size=8.4)
    colab = box(ax, 13.4, yc, 3.4, 1.8, "Colab A100", "train · infer", fc=BACKB_BG, ec=BACKB,
                tc=BACKB, title_size=12.5, sub_size=9)
    drive = box(ax, 19.0, yc, 3.9, 1.8, "Google Drive", "data · checkpoints", fc=DEPTH_BG,
                ec=DEPTH, tc=DEPTH, title_size=11.5, sub_size=8.6)
    shm = box(ax, 24.4, yc, 3.2, 1.8, "/dev/shm", "RAM disk ~83 GB", fc=NECK_BG, ec=NECK,
              tc=NECK, title_size=11.5, sub_size=8.6)
    arrow(ax, local["R"], gh["L"]); arrow(ax, gh["R"], colab["L"])
    arrow(ax, (colab["cx"], colab["T"][1]), (drive["cx"], drive["T"][1]), color=DEPTH, rad=-0.32, lw=1.8)
    arrow(ax, (drive["cx"], drive["B"][1]), (colab["cx"], colab["B"][1]), color=BACKB, rad=-0.32, lw=1.8)
    arrow(ax, drive["R"], shm["L"], color=NECK)
    ax.text((colab["cx"]+drive["cx"])/2, yc+1.95, "checkpoints · results", fontsize=8.2,
            color=MUTE, ha="center")
    ax.text((colab["cx"]+drive["cx"])/2, yc-1.95, "fetch data archives", fontsize=8.2,
            color=MUTE, ha="center", va="top")
    yb = 3.0
    raw = box(ax, 3.0, yb, 3.4, 1.7, "Raw .h264", "5k dishes · 181 GB", fc=INPUT_BG, ec=INPUT,
              tc=INPUT, title_size=11.5, sub_size=8.6)
    samp = box(ax, 8.8, yb, 3.7, 1.7, "Frame sample", "6 / camera · 292px", fc="#eef2f7",
               ec=DARK, tc=DARK, title_size=11.5, sub_size=8.6)
    arch = box(ax, 14.6, yb, 3.6, 1.7, ".tar.zst", "1.9 GB archives", fc=HEAD_BG, ec=HEAD,
               tc=HEAD, title_size=11.5, sub_size=8.6)
    upl = box(ax, 19.6, yb, 2.9, 1.7, "→ Drive", "upload", fc=DEPTH_BG, ec=DEPTH, tc=DEPTH,
              title_size=11.5, sub_size=8.6)
    arrow(ax, raw["R"], samp["L"]); arrow(ax, samp["R"], arch["L"]); arrow(ax, arch["R"], upl["L"], color=DEPTH)
    ax.text(0.4, 4.8, "Data prep (local)", fontsize=9.6, color=MUTE, ha="left", fontweight="bold")
    ax.text(0.4, 9.05, "Compute loop", fontsize=9.6, color=MUTE, ha="left", fontweight="bold")
    save(fig, "fig_pipeline")


# =============================== Fig 5 ====================================
def fig_resnet18():
    fig, ax = new_ax(13.2, 5.4, (0, 26.5), (0, 11))
    ax.text(0.4, 10.2, "Applied baseline — ResNet-18 calorie regressor", fontsize=15.5,
            color=INK, fontweight="bold", ha="left")
    ax.text(0.4, 9.45, "RGB (3ch) or RGB-D (4ch, depth as 4th channel)  ·  optional Food-101 auxiliary head",
            fontsize=10, color=MUTE, ha="left")
    yc = 5.2
    iw, ih = 3.5, 3.2
    ip = FancyBboxPatch((2.6-iw/2, yc-ih/2), iw, ih, boxstyle="round,pad=0.012,rounding_size=0.10",
                        lw=1.6, ec=INPUT, fc=INPUT_BG, zorder=3); ip.set_path_effects(SHADOW)
    ax.add_patch(ip)
    photo_glyph(ax, 2.6, yc+0.68, w=1.45, h=0.9)
    ax.text(2.6, yc-0.5, "Food image", ha="center", va="center", fontsize=12, color=INPUT,
            fontweight="bold", zorder=4)
    ax.text(2.6, yc-1.04, "RGB 3ch · RGB-D 4ch", ha="center", va="center", fontsize=8.0,
            color=MUTE, zorder=4)
    bb = fmap_stack(ax, 8.3, yc, "ResNet-18", "ImageNet-1K · 7×7×512 · 4-ch stem (RGB-D)",
                    n=4, base_h=2.7, base_w=0.5, gap=0.6)
    feat = box(ax, 13.2, yc, 2.2, 1.5, "512-d", "features", fc=NECK_BG, ec=NECK, tc=NECK,
               title_size=12, sub_size=9)
    arrow(ax, (2.6+iw/2, yc), (bb["L"][0]-0.15, yc)); arrow(ax, bb["R"], feat["L"])
    cal = box(ax, 20.0, yc+1.7, 5.9, 1.7, "Calorie head", "FC 512 → 128 → 1   →   kcal",
              fc=CAL_BG, ec=CAL, tc=CAL, title_size=12.5, sub_size=9.5)
    aux = box(ax, 20.0, yc-1.7, 5.9, 1.7, "Food-101 head (aux)", "FC 512 → 256 → 101",
              fc=HEAD_BG, ec=HEAD, tc=HEAD, title_size=11.5, sub_size=9.5)
    arrow(ax, feat["R"], cal["L"], color=CAL, rad=-0.16, lw=2.0)
    arrow(ax, feat["R"], aux["L"], color=FANC, rad=0.16, lw=1.8, ls=(0, (4, 3)))
    ax.text(0.4, 1.3, "Target:  regress  log1p(calories),  decode with  expm1   ·   loss: SmoothL1",
            fontsize=10, color=INK, ha="left")
    save(fig, "fig_resnet18")


# ====================== result plots (shared style) =======================
def fig_results_metrics():
    runs = ["RGB-D (primary)", "RGB + Food-101 aux"]
    mae  = [79.10, 80.31]
    rmse = [125.89, 127.35]
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); fig.patch.set_facecolor(PAGE)
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x-w/2, mae, w, label="Test MAE", color=C_RGBD, zorder=3, edgecolor="white", lw=0.8)
    b2 = ax.bar(x+w/2, rmse, w, label="Test RMSE", color=HEAD, zorder=3, edgecolor="white", lw=0.8)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{b.get_height():.1f}",
                ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    style_axes(ax, "Test error — calorie estimation (kcal)")
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=10, color=INK)
    ax.set_ylabel("kcal", fontsize=10, color=INK); ax.set_ylim(0, 145)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, "fig_results_metrics")


def fig_results_scatter():
    t, p = read_preds(RGBD_RUN)
    lim = 1100  # axis cap for readability; 1 dish over-predicts off-scale
    a1lab = "predicted calories (kcal)"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6)); fig.patch.set_facecolor(PAGE)
    a1.plot([0, lim], [0, lim], ls="--", color=MUTE, lw=1.3, zorder=2, label="perfect (y = x)")
    a1.scatter(t, p, s=14, color=C_RGBD, alpha=0.45, edgecolor="none", zorder=3)
    style_axes(a1, "RGB-D predictions vs. ground truth")
    a1.set_xlabel("true calories (kcal)", fontsize=10, color=INK)
    a1.set_ylabel(a1lab, fontsize=10, color=INK)
    a1.set_xlim(0, lim); a1.set_ylim(0, lim); a1.legend(frameon=False, fontsize=9.5, loc="upper left")
    a1.text(0.97, 0.06, "MAE 79.1 · RMSE 125.9 kcal", transform=a1.transAxes, ha="right",
            fontsize=9.5, color=INK, fontweight="bold")
    a1.text(0.97, 0.14, "(1 dish over-predicts off-scale)", transform=a1.transAxes, ha="right",
            fontsize=8.0, color=MUTE)
    res = p - t
    a2.axhline(0, color=MUTE, ls="--", lw=1.3, zorder=2)
    a2.scatter(t, res, s=14, color=HEAD, alpha=0.45, edgecolor="none", zorder=3)
    style_axes(a2, "Residuals  (pred − true)")
    a2.grid(axis="both", color=GRIDC, lw=1.0)
    a2.set_xlabel("true calories (kcal)", fontsize=10, color=INK)
    a2.set_ylabel("error (kcal)", fontsize=10, color=INK)
    a2.set_xlim(0, lim); a2.set_ylim(-450, 650)
    a2.text(0.97, 0.06, "mean bias −24.9 kcal", transform=a2.transAxes, ha="right",
            fontsize=9.5, color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_results_scatter")


def fig_results_errbin():
    names, v_rgbd = read_bins(RGBD_RUN)
    _, v_rgb = read_bins(RGB_RUN)
    fig, ax = plt.subplots(figsize=(7.6, 4.2)); fig.patch.set_facecolor(PAGE)
    x = np.arange(len(names)); w = 0.4
    ax.bar(x-w/2, v_rgbd, w, label="RGB-D", color=C_RGBD, zorder=3, edgecolor="white", lw=0.7)
    ax.bar(x+w/2, v_rgb, w, label="RGB", color=HEAD, zorder=3, edgecolor="white", lw=0.7)
    style_axes(ax, "Mean absolute error by true-calorie level")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.6, color=INK, rotation=0)
    ax.set_xlabel("true calorie bin (kcal)", fontsize=10, color=INK)
    ax.set_ylabel("MAE (kcal)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, "fig_results_errbin")


def fig_results_cdf():
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); fig.patch.set_facecolor(PAGE)
    for run, col, lab in ((RGBD_RUN, C_RGBD, "RGB-D"), (RGB_RUN, HEAD, "RGB")):
        t, p = read_preds(run); e = np.sort(np.abs(p-t))
        cdf = np.arange(1, len(e)+1)/len(e)*100
        ax.plot(e, cdf, color=col, lw=2.4, label=lab, zorder=3)
    ax.axvline(100, color=MUTE, ls="--", lw=1.2, zorder=2)
    ax.text(104, 12, "100 kcal", color=MUTE, fontsize=9, rotation=90, va="bottom")
    ax.scatter([100], [71.8], color=C_RGBD, zorder=5, s=30)
    ax.annotate("71.8% within 100 kcal", (100, 71.8), (150, 58), fontsize=9.5, color=INK,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    style_axes(ax, "Absolute-error CDF (test set)")
    ax.set_xlabel("absolute error (kcal)", fontsize=10, color=INK)
    ax.set_ylabel("% of dishes ≤ error", fontsize=10, color=INK)
    ax.set_xlim(0, 400); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    save(fig, "fig_results_cdf")


def fig_results_examples():
    t, p = read_preds(RGBD_RUN); e = np.abs(p-t); order = np.argsort(e)
    pick = list(order[:5]) + list(order[-5:])          # 5 best + 5 worst
    tt, pp, ee = t[pick], p[pick], e[pick]
    y = np.arange(len(pick))
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); fig.patch.set_facecolor(PAGE)
    for yi, (a, b) in enumerate(zip(tt, pp)):
        ax.plot([a, b], [yi, yi], color="#cbd5e1", lw=2.6, zorder=2, solid_capstyle="round")
    ax.scatter(tt, y, s=70, color=MUTE, zorder=3, label="ground truth")
    ax.scatter(pp, y, s=70, color=C_RGBD, zorder=4, label="predicted")
    for yi, (a, b, err) in enumerate(zip(tt, pp, ee)):
        ax.text(max(a, b)+30, yi, f"err {err:.0f}", va="center", ha="left", fontsize=8.2,
                color=INK, zorder=5)
    style_axes(ax, "Best (top) and worst (bottom) RGB-D predictions")
    ax.grid(axis="x", color=GRIDC, lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(["best"]*5 + ["worst"]*5, fontsize=9, color=INK)
    ax.axhline(4.5, color="#9ca3af", lw=1.0, ls=":")
    ax.set_xlabel("calories (kcal)", fontsize=10, color=INK)
    ax.set_xlim(-40, 1950); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "fig_results_examples")


def fig_data():
    t, _ = read_preds(RGBD_RUN)                     # 507 test-set target calories
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                 gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor(PAGE)
    a1.hist(t, bins=24, color=C_RGBD, alpha=0.88, edgecolor="white", linewidth=0.7, zorder=3)
    med = float(np.median(t))
    a1.axvline(med, color=CAL, ls="--", lw=1.7, zorder=4)
    a1.text(med+18, a1.get_ylim()[1]*0.9, f"median {med:.0f} kcal", color=CAL, fontsize=9.2,
            fontweight="bold")
    style_axes(a1, "Calorie distribution (test set)")
    a1.set_xlabel("calories (kcal)", fontsize=10, color=INK)
    a1.set_ylabel("# dishes", fontsize=10, color=INK)
    a1.text(0.97, 0.78, "long high-calorie tail", transform=a1.transAxes, ha="right",
            fontsize=8.6, color=MUTE)
    names, counts, cols = ["train", "val", "test"], [2160, 240, 507], [C_RGBD, NECK, HEAD]
    bars = a2.bar(names, counts, color=cols, zorder=3, edgecolor="white", linewidth=0.8)
    for b, c in zip(bars, counts):
        a2.text(b.get_x()+b.get_width()/2, c+30, str(c), ha="center", va="bottom",
                fontsize=9.5, color=INK, fontweight="bold")
    style_axes(a2, "Local subset split")
    a2.set_ylabel("# dishes", fontsize=10, color=INK); a2.set_ylim(0, 2450)
    fig.tight_layout()
    save(fig, "fig_data")


# ==================== concept figures (ex-mermaid) ========================
def card(ax, cx, cy, w, h, title, lines, ec, bg, tc, title_size=13):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       lw=1.8, ec=ec, fc=bg, zorder=3); p.set_path_effects(SHADOW); ax.add_patch(p)
    ax.text(cx, cy+h/2-0.55, title, ha="center", va="center", fontsize=title_size, color=tc,
            fontweight="bold", zorder=4)
    ty = cy + h/2 - 1.45
    for ln in lines:
        ax.text(cx-w/2+0.55, ty, "•  " + ln, ha="left", va="center", fontsize=9.6, color=INK, zorder=4)
        ty -= 0.74


def fig_overview():
    fig, ax = new_ax(12.6, 6.8, (0, 25), (0, 13.6))
    ax.text(0.4, 13.0, "Two complementary tracks", fontsize=16.5, color=INK, fontweight="bold", ha="left")
    box(ax, 12.5, 11.3, 13.5, 1.7, "Nutrition5k dataset",
        "5k dishes · overhead RGB-D · 4× side-angle video · official train/test splits",
        fc=NECK_BG, ec=NECK, tc=NECK, title_size=13, sub_size=9.0)
    card(ax, 6.2, 6.3, 10.6, 6.3, "Track B — Paper reproduction",
         ["InceptionV3 multi-task network", "Exp 1–4  +  DPF-Nutrition extension",
          "official splits · fixed 82% leak", "Colab A100 · Google Drive", "metric:  PMAE %"],
         BACKB, BACKB_BG, BACKB)
    card(ax, 18.8, 6.3, 10.6, 6.3, "Track A — Applied baseline + demo",
         ["ResNet-18 RGB / RGB-D regressor", "+ optional Food-101 aux head",
          "storage-limited local subset", "Gradio web app on HF Spaces", "metric:  kcal MAE"],
         HEAD, HEAD_BG, HEAD)
    arrow(ax, (9.8, 10.45), (6.2, 9.45), color=BACKB, rad=0.10)
    arrow(ax, (15.2, 10.45), (18.8, 9.45), color=HEAD, rad=-0.10)
    box(ax, 12.5, 0.85, 13.8, 1.4, "Unified study",
        "research fidelity  ×  practical deployment", fc="#eef2f7", ec=DARK, tc=DARK,
        title_size=12, sub_size=9.2)
    arrow(ax, (6.2, 3.15), (10.9, 1.6), color=BACKB, rad=0.14)
    arrow(ax, (18.8, 3.15), (14.1, 1.6), color=HEAD, rad=-0.14)
    save(fig, "fig_overview")


def fig_gap():
    fig, ax = new_ax(13.2, 6.2, (0, 27), (0, 12.2))
    ax.text(0.4, 11.6, "Why the reproduction gap?", fontsize=15.5, color=INK, fontweight="bold", ha="left")
    root = box(ax, 3.0, 6.0, 3.6, 1.9, "Reproduction\ngap", fc=CAL_BG, ec=CAL, tc=CAL, title_size=12.5)
    cats = [("Public pretraining", ["JFT-300M not public", "Food2K helps only slightly"]),
            ("Input / cache", ["6-frame side-angle sampling", "3 missing DPF depth dishes"]),
            ("Implementation", ["InceptionV3 ≠ InceptionV2 / JFT", "Depth Anything V2 ≠ paper's"]),
            ("Dataset scale", ["macros weakly visible in 2D", "too small for bigger backbones"])]
    ys = np.linspace(10.2, 1.8, 4)
    for (name, leaves), yy in zip(cats, ys):
        cb = box(ax, 10.2, yy, 4.4, 1.45, name, fc=BACKB_BG, ec=BACKB, tc=BACKB, title_size=11.5)
        arrow(ax, root["R"], cb["L"], color=FANC, lw=1.9, rad=0.10*np.sign(yy-6))
        for li, leaf in enumerate(leaves):
            ly = yy + (0.85 if li == 0 else -0.85)
            lb = box(ax, 20.0, ly, 7.8, 0.92, leaf, fc="#ffffff", ec="#94a3b8", tc=INK,
                     title_size=9.2, bold=False, shadow=False, round_size=0.16)
            arrow(ax, cb["R"], lb["L"], color=FANC, lw=1.4, rad=0.06*np.sign(ly-yy))
    save(fig, "fig_gap")


def fig_inference():
    fig, ax = new_ax(13.0, 3.7, (0, 26), (0, 7.3))
    ax.text(0.4, 6.7, "Web-app inference flow", fontsize=14.5, color=INK, fontweight="bold", ha="left")
    yc = 3.7
    u = box(ax, 3.0, yc, 3.6, 1.7, "Upload image", "food photo", fc=INPUT_BG, ec=INPUT, tc=INPUT,
            title_size=11.5, sub_size=9)
    pp = box(ax, 8.4, yc, 3.6, 1.7, "Preprocess", "resize · ImageNet norm", fc="#eef2f7", ec=DARK,
             tc=DARK, title_size=11.5, sub_size=8.4)
    m = box(ax, 14.0, yc, 3.6, 1.7, "Checkpoint", "RGB or RGB-D", fc=BACKB_BG, ec=BACKB, tc=BACKB,
            title_size=11.5, sub_size=9)
    k = box(ax, 20.4, yc, 4.2, 1.7, "Predicted kcal", "+ optional top-k class", fc=CAL_BG, ec=CAL,
            tc=CAL, title_size=12, sub_size=9)
    arrow(ax, u["R"], pp["L"]); arrow(ax, pp["R"], m["L"]); arrow(ax, m["R"], k["L"], color=CAL)
    md = box(ax, 11.2, 1.0, 5.2, 1.3, "MiDaS / heuristic depth", "when RGB-D has no real depth",
             fc=DEPTH_BG, ec=DEPTH, tc=DEPTH, title_size=10.5, sub_size=8.0)
    arrow(ax, (pp["cx"], pp["B"][1]), (md["L"][0]-0.05, md["cy"]), color=DEPTH, ls=(0, (4, 3)),
          rad=-0.25, lw=1.6)
    arrow(ax, (md["R"][0], md["cy"]), (m["cx"], m["B"][1]), color=DEPTH, ls=(0, (4, 3)),
          rad=-0.25, lw=1.6)
    save(fig, "fig_inference")


def fig_glance():
    fig, ax = new_ax(12.6, 5.4, (0, 25), (0, 11))
    ax.text(0.4, 10.4, "Results at a glance", fontsize=16, color=INK, fontweight="bold", ha="left")

    def panel(cx, header, ec, bg, stats):
        p = FancyBboxPatch((cx-5.4, 0.7), 10.8, 8.4, boxstyle="round,pad=0.02,rounding_size=0.12",
                           lw=1.8, ec=ec, fc=bg, zorder=2); p.set_path_effects(SHADOW); ax.add_patch(p)
        ax.text(cx, 8.4, header, ha="center", va="center", fontsize=11.5, color=ec,
                fontweight="bold", zorder=4)
        ys = np.linspace(6.6, 1.7, len(stats))
        for (big, small, col), yy in zip(stats, ys):
            ax.text(cx-4.7, yy, big, ha="left", va="center", fontsize=22, color=col,
                    fontweight="bold", zorder=4)
            ax.text(cx-1.0, yy, small, ha="left", va="center", fontsize=9.4, color=INK, zorder=4)

    panel(6.1, "Track B · paper reproduction  (PMAE %)", BACKB, BACKB_BG,
          [("22.0%", "Exp 3 RGB-D calories\n(paper 18.8%)", BACKB),
           ("17.8%", "Exp 3 mass — beats paper 18.9%", DEPTH),
           ("22.67%", "DPF mean PMAE\n(paper 17.8%)", BACKB)])
    panel(18.9, "Track A · applied baseline  (kcal)", HEAD, HEAD_BG,
          [("79.1", "RGB-D test MAE (kcal)", HEAD),
           ("71.8%", "dishes within 100 kcal", HEAD)])
    save(fig, "fig_glance")


if __name__ == "__main__":
    fig_multitask(); fig_exp4(); fig_dpf(); fig_pipeline(); fig_resnet18()
    fig_results_metrics(); fig_results_scatter(); fig_results_errbin()
    fig_results_cdf(); fig_results_examples(); fig_data()
    fig_overview(); fig_gap(); fig_inference(); fig_glance()
    print("done")
