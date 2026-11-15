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