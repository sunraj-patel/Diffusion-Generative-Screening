"""Shared matplotlib style: warm ivory canvas, slate ink, clay/blue/green accents."""
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

INK, MUTED, RULE = "#000000", "#000000", "#CCCCCC"
BG, PANEL = "#FFFFFF", "#FFFFFF"
CLAY, BLUE, GREEN = "#D97757", "#6A9BCC", "#788C5D"
PLUM, GOLD, TEAL = "#8E6BA8", "#C9A227", "#4F8A8B"
CYCLE = [CLAY, BLUE, GREEN, PLUM, GOLD, TEAL]


def _pick(cands):
    have = {f.name for f in fm.fontManager.ttflist}
    return next((c for c in cands if c in have), cands[-1])


SANS = _pick(["Inter", "Arial", "Helvetica Neue", "DejaVu Sans"])
SERIF = _pick(["Tiempos Headline", "Georgia", "Times New Roman", "DejaVu Serif"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.facecolor": PANEL, "axes.edgecolor": RULE,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.linewidth": 0.9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelpad": 6,
        "font.family": SANS, "font.size": 10, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "grid.color": RULE, "grid.linewidth": 0.8, "grid.alpha": 1.0,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "legend.labelcolor": INK,
        "lines.linewidth": 1.8, "lines.markersize": 5.5,
        "savefig.dpi": 200, "figure.dpi": 100,
    })


def header(fig, title, subtitle=None):
    """Left-aligned serif title + muted subtitle above the axes."""
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top",
                 fontfamily=SERIF, fontsize=16, fontweight="normal", color=INK)
    if subtitle:
        fig.text(0.01, 0.995 - 0.055, subtitle, ha="left", va="top",
                 fontsize=9.5, color=MUTED)


def panel_label(ax, s):
    ax.text(-0.02, 1.0, "", transform=ax.transAxes)                       
    ax.set_title(s + "   " + ax.get_title(), loc="left")


def finish(ax, grid="y"):
    ax.grid(True, axis=grid if grid != "both" else "both")
    ax.tick_params(length=3, width=0.8, color=RULE)
