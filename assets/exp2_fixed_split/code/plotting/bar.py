import pandas as pd
import matplotlib.pyplot as plt

from plotting.style import set_plot_style
from plotting.save import save_figure


def plot_bar(
    csv_path,
    x,
    y,
    out_dir,
    output_name,
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=None,
    value_labels=True,
    colors=None,
):
    set_plot_style()
    df = pd.read_csv(csv_path)

    if x not in df.columns:
        raise ValueError(f"CSV 中不存在列：{x}")

    if y not in df.columns:
        raise ValueError(f"CSV 中不存在列：{y}")

    if colors is None:
        colors = [
            "#4C72B0",
            "#DD8452",
            "#55A868",
            "#C44E52",
            "#8172B3",
            "#937860",
            "#DA8BC3",
            "#8C8C8C",
        ]

    bar_colors = [colors[i % len(colors)] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    bars = ax.bar(
        df[x],
        df[y],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    if value_labels:
        for bar, value in zip(bars, df[y]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    # 坐标轴
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8,
    )

    fig.tight_layout()

    save_figure(fig, out_dir, output_name)