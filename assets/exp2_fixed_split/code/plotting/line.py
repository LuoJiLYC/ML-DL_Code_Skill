import pandas as pd

from plotting.style import COLORS, set_plot_style

import matplotlib.pyplot as plt

from plotting.save import save_figure


def plot_line(
    csv_path,
    x,
    ys,
    labels,
    xlabel,
    ylabel,
    out_dir,
    output_name,
    title=None,
    ylim=None,
    markers=None,
):
    """根据 CSV 绘制通用折线图。"""

    set_plot_style()
    df = pd.read_csv(csv_path)

    # 检查 x 列
    if x not in df.columns:
        raise ValueError(f"CSV 中不存在 x 列：{x}")

    # 检查参数长度
    if len(ys) != len(labels):
        raise ValueError("ys 和 labels 的长度必须一致")

    # 默认所有折线都带圆点
    if markers is None:
        markers = ["o"] * len(ys)

    if len(ys) != len(markers):
        raise ValueError("ys 和 markers 的长度必须一致")

    # 检查 y 列
    for y in ys:
        if y not in df.columns:
            raise ValueError(f"CSV 中不存在 y 列：{y}")

    # 创建图
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # 绘制每条折线
    for i, (y, label, marker) in enumerate(zip(ys, labels, markers)):
        plot_df = df[[x, y]].dropna()

        ax.plot(
            plot_df[x],
            plot_df[y],
            color=COLORS[i % len(COLORS)],
            linewidth=1.5,
            marker=marker,
            markersize=3,
            label=label,
        )

    # 坐标轴
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 标题
    if title is not None:
        ax.set_title(title)

    # y 轴范围
    if ylim is not None:
        ax.set_ylim(ylim)

    # 图例
    ax.legend()

    # 自动调整布局
    fig.tight_layout()

    # 保存
    save_figure(fig, out_dir, output_name)



import pandas as pd
import matplotlib.pyplot as plt

from plotting.style import set_plot_style
from plotting.save import save_figure


def plot_line(csv_path, x, ys, labels, out_dir, output_name, xlabel=None, ylabel=None, title=None, ylim=None, markers=None):
    set_plot_style()
    df = pd.read_csv(csv_path)

    if markers is None:
        markers = [None] * len(ys)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for y, label, marker in zip(ys, labels, markers):
        ax.plot(df[x], df[y], label=label, marker=marker)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, output_name)


def plot_line_by_row_models(
    csv_path,
    out_dir,
    output_name,
    model_col="model",
    xlabel="Training Set Size",
    ylabel="Accuracy",
    title=None,
    ylim=None,
    markers=None,
    colors=None,
):
    """
    适用于如下 CSV:
    model,2000,4000,6000
    CatBoost,0.5585,0.5755,0.5750
    MLP,0.5665,0.5750,0.5815
    """

    set_plot_style()
    df = pd.read_csv(csv_path)

    if model_col not in df.columns:
        raise ValueError(f"CSV 中不存在列: {model_col}")

    x_labels = [c for c in df.columns if c != model_col]
    x_values = [int(v) for v in x_labels]

    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

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

    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    for i, (_, row) in enumerate(df.iterrows()):
        model_name = row[model_col]
        y_values = [row[c] for c in x_labels]

        ax.plot(
            x_values,
            y_values,
            label=model_name,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(x_values)
    ax.set_xticklabels([str(v) for v in x_values])

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, width=0.8)

    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, out_dir, output_name)