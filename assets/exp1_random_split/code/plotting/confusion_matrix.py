import json
import numpy as np
import matplotlib.pyplot as plt

from plotting.style import set_plot_style
from plotting.save import save_figure


def plot_confusion_matrix(
    json_path,
    out_dir,
    output_name,
    labels=None,
    xlabel="Predicted label",
    ylabel="True label",
    title=None,
    normalize=False,
):
    """从 result.json 中读取 confusion_matrix 并绘图。"""

    set_plot_style()

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    cm = np.array(result["metrics"]["confusion_matrix"])

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion_matrix 必须是方阵")

    if labels is None:
        labels = list(range(cm.shape[0]))

    if len(labels) != cm.shape[0]:
        raise ValueError("labels 数量必须与混淆矩阵类别数一致")

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(
            cm,
            row_sum,
            out=np.zeros_like(cm, dtype=float),
            where=row_sum != 0,
        )
    else:
        plot_cm = cm

    fig, ax = plt.subplots(figsize=(3.0, 2.8))

    im = ax.imshow(plot_cm, cmap="Blues")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 去掉坐标轴四周边框线
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 去掉坐标轴刻度线，但保留刻度文字
    ax.tick_params(axis="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 去掉 colorbar 外框线
    cbar.outline.set_visible(False)

    # 去掉 colorbar 的刻度线，但保留数字
    cbar.ax.tick_params(length=0)


    if title is not None:
        ax.set_title(title)

    threshold = plot_cm.max() / 2 if plot_cm.size else 0

    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            if normalize:
                text = f"{plot_cm[i, j]:.2f}"
            else:
                text = str(cm[i, j])

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if plot_cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    save_figure(fig, out_dir, output_name)