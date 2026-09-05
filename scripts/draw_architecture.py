"""Draw the TabPFN-M architecture figure (docs/figures/tabpfn_m_architecture.{png,svg})."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OBS, MISS, TGT, THINK = "#4C78A8", "#E6E6E6", "#F58518", "#BBBBBB"
C1, C2, C3 = "#2CA02C", "#7B3FB5", "#D62728"
TXT = "#222222"

fig, ax = plt.subplots(figsize=(17, 9.2))
ax.set_xlim(0, 17); ax.set_ylim(0.3, 9.0); ax.axis("off")


def box(x, y, w, h, title=None, color="#888888", lw=1.6, fill="white", title_size=10.5, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", ec=color, fc=fill, lw=lw,
                       ls="--" if dashed else "-")
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=title_size, weight="bold", color=color)


def arrow(x0, y0, x1, y1, color="#444444", lw=1.6, style="-|>", ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 linestyle=ls, alpha=alpha, shrinkA=2, shrinkB=2))


def label(x, y, s, size=9, color=TXT, ha="center", va="center", style="normal", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, style=style, weight=weight)


# ------------------------------------------------------------------ 1. input table
bx, by = 0.35, 4.3
box(bx, by, 3.3, 4.5, "Input table (NaN = not measured)", "#555555", title_size=9.4)
cell = 0.3
sources = [("lab A", [0, 1, 2], "#9ECAE1"), ("lab B", [0, 3, 4, 5], "#A1D99B"), ("lab C", [1, 2, 6], "#FDD0A2")]
rows = []
for name, feats, col in sources:
    rows += [(name, feats, col)] * 2
x0, y0 = bx + 0.75, by + 3.2
for r, (name, feats, col) in enumerate(rows):
    y = y0 - r * (cell + 0.06)
    ax.add_patch(Rectangle((x0 - 0.42, y), 0.32, cell, fc=col, ec="none"))
    if r % 2 == 0:
        label(x0 - 0.62, y - 0.03, name, 8, ha="right")
    for f in range(7):
        obs = f in feats
        ax.add_patch(Rectangle((x0 + f * (cell + 0.03), y), cell, cell, fc=OBS if obs else MISS,
                               ec="white" if obs else "#AAAAAA", lw=0.8, hatch=None if obs else "////"))
label(x0 + 3.5 * (cell + 0.03) - 0.02, y0 + cell + 0.16, "features 1 … 7", 8.5)
label(x0 + 3.5 * (cell + 0.03) - 0.02, y0 - 6 * (cell + 0.06) - 0.05, "rows keep 3 or 4 of 7 features,\nin blocks by source", 8.5, style="italic")
# legend
ax.add_patch(Rectangle((bx + 0.25, by + 0.28), 0.22, 0.22, fc=OBS)); label(bx + 0.55, by + 0.39, "observed", 8, ha="left")
ax.add_patch(Rectangle((bx + 1.55, by + 0.28), 0.22, 0.22, fc=MISS, ec="#AAAAAA", hatch="////")); label(bx + 1.85, by + 0.39, "missing (NaN)", 8, ha="left")

# ------------------------------------------------------------------ 2. context + encoder
arrow(bx + 3.3, 6.6, 4.35, 6.6)
box(4.35, 5.55, 2.6, 2.1, "TabPFN encoder", "#555555")
label(5.65, 6.75, "mean-impute NaN cells\n+ NaN indicator per cell\ngroup 3 features → 1 token\nappend target token", 8.6)
label(5.65, 5.72, "pretrained, unchanged", 8, style="italic", color="#666666")

box(4.35, 3.05, 2.6, 2.1, "Missingness context", C1, dashed=True)
label(5.65, 4.25, "from the raw NaN pattern:\ntoken mask  (row, group)\nmissing groups  (row, group)\nJaccard(m$_i$, m$_j$)  (row, row)", 8.6)
label(5.65, 3.22, "no parameters", 8, style="italic", color="#666666")
arrow(bx + 3.3, 5.0, 4.35, 4.1, color=C1)

# ------------------------------------------------------------------ 3. token grid + absence vector
arrow(6.95, 6.6, 7.8, 6.6)
box(7.8, 5.05, 2.75, 3.75, "Token grid", "#555555")
gx, gy = 8.15, 7.7
for r in range(6):
    y = gy - r * 0.37
    feats = rows[r][1]
    for g in range(3):
        present = any(f in feats for f in range(3 * g, 3 * g + 3))
        ax.add_patch(Rectangle((gx + g * 0.5, y), 0.42, 0.34, fc=OBS if present else MISS, ec="#AAAAAA" if not present else "white", lw=0.8))
        if not present:
            label(gx + g * 0.5 + 0.21, y + 0.17, "+a", 8, color=C1, weight="bold")
    ax.add_patch(Rectangle((gx + 3 * 0.5 + 0.1, y), 0.42, 0.34, fc=TGT, ec="white"))
label(gx + 0.75, gy + 0.52, "3 feature groups", 8)
label(gx + 1.81, gy + 0.52, "target", 8)
label(9.17, 5.27, "1b  +a : learned absence vector on fully missing group tokens", 7.9, color=C1)
label(9.17, 5.62, "64 thinking rows prepended (not drawn)", 7.8, style="italic", color="#666666")
arrow(6.95, 4.1, 8.7, 5.05, color=C1, ls="--")

# ------------------------------------------------------------------ 4. layer stack
arrow(10.55, 6.9, 11.25, 6.9)
LX, LY, LW, LH = 11.25, 1.55, 5.4, 7.25
box(LX, LY, LW, LH, "× 18 layers   (pretrained q, k, v, MLP weights reused)", "#555555", title_size=10)
# 1a feature attention
box(LX + 0.2, LY + 4.65, LW - 0.4, 1.95, "1a  Observed-only feature attention", C1, title_size=9.6)
fx, fy = LX + 0.75, LY + 5.3
toks = [OBS, MISS, OBS, TGT]
for i, c in enumerate(toks):
    ax.add_patch(Rectangle((fx + i * 0.95, fy), 0.55, 0.42, fc=c, ec="#AAAAAA" if c == MISS else "white"))
label(fx + 0.27, fy + 0.58, "q", 8); label(fx + 0.95 + 0.27, fy + 0.58, "missing", 7.5, color="#888888")
for i in (2, 3):
    arrow(fx + 0.55, fy + 0.21 + (0.08 if i == 2 else -0.08), fx + i * 0.95, fy + 0.21 + (0.08 if i == 2 else -0.08), color=C1, lw=1.4, style="-|>")
ax.plot([fx + 0.95 + 0.1, fx + 0.95 + 0.45], [fy + 0.05, fy + 0.37], color=C2 if False else "#B00020", lw=2)
ax.plot([fx + 0.95 + 0.1, fx + 0.95 + 0.45], [fy + 0.37, fy + 0.05], color="#B00020", lw=2)
label(LX + LW / 2 + 0.05, LY + 4.92, "a row's tokens attend only to its observed groups (+ target);\nmissing tokens are masked as keys, still updated as queries", 8.2)
# 2 item attention
box(LX + 0.2, LY + 1.65, LW - 0.4, 2.85, "2  Pattern-biased item attention (rows of one column)", C2, title_size=9.2)
ix, iy = LX + 0.85, LY + 3.45
train = [("lab A", "#9ECAE1", 1.0), ("lab A", "#9ECAE1", 1.0), ("lab B", "#A1D99B", 0.2), ("lab C", "#FDD0A2", 0.5)]
for i, (n, c, s) in enumerate(train):
    y = iy - i * 0.45
    ax.add_patch(Rectangle((ix + 2.6, y), 0.5, 0.34, fc=OBS, ec="white"))
    ax.add_patch(Rectangle((ix + 3.2, y), 0.18, 0.34, fc=c, ec="none"))
    label(ix + 3.5, y + 0.17, f"J = {s:.1f}", 7.8, ha="left")
    arrow(ix + 0.55, iy - 0.7 + 0.17, ix + 2.6, y + 0.17, color=C2, lw=0.6 + 2.2 * s, alpha=0.45 + 0.55 * s)
ax.add_patch(Rectangle((ix, iy - 0.7), 0.5, 0.34, fc=OBS, ec="white"))
ax.add_patch(Rectangle((ix - 0.25, iy - 0.7), 0.18, 0.34, fc="#9ECAE1", ec="none"))
label(ix + 0.25, iy - 0.95, "test row (lab A)", 7.8)
label(ix + 2.85, iy + 0.5, "training rows", 7.8)
label(LX + LW / 2 + 0.05, LY + 1.85, r"score$_{ij}$ += $\alpha_\ell$ · Jaccard(m$_i$, m$_j$)     one scalar $\alpha_\ell$ per layer, init 0", 8.4, color=C2)
# MLP
box(LX + 0.2, LY + 0.25, LW - 0.4, 1.15, None, "#888888")
label(LX + LW / 2, LY + 0.82, "MLP  (pretrained)", 9.5, color="#555555")
label(LX + LW / 2, LY + 0.42, "post-norm residual blocks as in TabPFN", 7.8, style="italic", color="#666666")
arrow(6.95, 3.6, LX + 0.2, LY + 5.6, color=C1, ls="--", alpha=0.8)
arrow(6.95, 3.4, LX + 0.2, LY + 2.9, color=C2, ls="--", alpha=0.8)
label(8.9, 3.95, "context shared with every layer", 7.8, color="#666666", style="italic")

# ------------------------------------------------------------------ 5. outputs
arrow(LX + LW, 7.9, LX + LW + 0.3, 7.9); 
label(LX + LW + 0.32, 8.35, "target tokens of\ntest rows → decoder\n→ prediction", 8.4, ha="left", color=TXT)
arrow(LX + LW, 3.0, LX + LW + 0.3, 3.0, color=C3)
label(LX + LW + 0.32, 3.45, "3  reconstruction head\nfeature tokens → hidden\ncell values (train only)", 8.2, ha="left", color=C3)

# ------------------------------------------------------------------ 6. training-only augmentation
box(0.35, 0.45, 10.2, 2.4, "3  Training only: block-missingness augmentation + masked-cell reconstruction", C3, title_size=10, dashed=True)
ax_x = 0.7
label(ax_x, 2.35, "pseudo-sources: split the rows of each training table into 2–6 groups, each keeps a random\n30–90 % of the features and hides the rest (+ 3 % random cells)", 8.6, ha="left", va="top")
label(ax_x, 1.5, "reconstruction: hide 15 % of the observed cells, predict their standardised values from the final\nfeature tokens with a linear head; smooth-L1 loss added to the TabPFN target loss", 8.6, ha="left", va="top")
label(ax_x, 0.68, "new parameters: 18 α, one 192-vector a, a 192×3 head.  With all switches off the model equals the pretrained TabPFN exactly.", 8.2, ha="left", style="italic", color="#444444")
arrow(3.9, 2.85, 3.9, 4.3, color=C3, ls="--")

fig.suptitle("TabPFN-M: three additions to a pretrained TabPFN for rows with different observed features", fontsize=13, y=0.985, weight="bold")
for ext in ("png", "svg"):
    fig.savefig(f"docs/figures/tabpfn_m_architecture.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight", facecolor="white")
print("saved")
