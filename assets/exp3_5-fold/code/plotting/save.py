import os
import matplotlib.pyplot as plt


def save_figure(fig, out_dir, name, dpi=600):
    """保存图为 PNG、PDF 和 SVG。"""

    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, f"{name}.svg"), bbox_inches="tight", facecolor="white")

    plt.close(fig)