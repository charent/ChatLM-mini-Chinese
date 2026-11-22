"""训练日志 loss 曲线绘制。"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def str_to_timestamp(string: str) -> float:
    """将日志中的时间戳字符串（可带方括号）转换为 float 时间戳。"""
    date_fmt = "%Y-%m-%d %H:%M:%S.%f"
    string = string.replace("[", "").replace("]", "")

    return datetime.strptime(string, date_fmt).timestamp()


def plot_training_loss(
    log_file: str,
    start_date: str,
    end_date: str,
    pic_save_to_file: str | None = None,
) -> None:
    """将 log 日志中记录的 loss 画图，按需保存到文件。

    由于 log 日志打印内容较多，需要指定要打印 loss 的开始时间和结束时间。

    example:
        >>> plot_training_loss('./artifacts/logs/chat_trainer.log', '[2023-10-01 08:44:39.303]', '[2023-10-01 11:29:12.376]')
    """
    start_timestamp = str_to_timestamp(start_date)
    end_timestamp = str_to_timestamp(end_date)

    loss_list = []
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            if "training loss: epoch:" in line:
                line = line.split(" ")
                date = " ".join(line[0:2])
                if str_to_timestamp(date) < start_timestamp:
                    continue

                if str_to_timestamp(date) > end_timestamp:
                    break

                if len(line) != 9:
                    continue

                epoch = line[5][6:-1]  # 'epoch:0,'
                step = line[6][5:-1]  # 'step:0,'
                loss = float(line[7][5:-1])  # 'loss:0.11086619377136231\n'
                device = line[8][7:-1]
                loss_list.append([epoch, step, loss, device])

    df = pd.DataFrame(loss_list, columns=["epoch", "step", "loss", "device"])

    # 多项式拟合
    x = list(range(0, len(df["loss"])))
    x_range = np.arange(0, len(df["loss"]), step=0.005)
    fit3 = np.polyfit(x, df["loss"], 3)
    p1d = np.poly1d(fit3)
    y_fit = p1d(x_range)

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(df["loss"], "g", label="loss")
    plt.plot(x_range, y_fit, "r", label="fit loss")
    plt.ylabel("loss")
    plt.xlabel("sampling step")
    plt.legend()

    if pic_save_to_file is not None:
        plt.savefig(pic_save_to_file)

    plt.show()


# 兼容旧名称（原函数名拼写错误）
plot_traing_loss = plot_training_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制训练日志中的 loss 曲线")
    parser.add_argument("log_file", help="训练日志文件路径")
    parser.add_argument("start_date", help="起始时间，如 '[2023-10-01 08:44:39.303]'")
    parser.add_argument("end_date", help="结束时间，如 '[2023-10-01 11:29:12.376]'")
    parser.add_argument("--save-path", default=None, help="图片保存路径，不传则仅显示")
    args = parser.parse_args()

    plot_training_loss(args.log_file, args.start_date, args.end_date, pic_save_to_file=args.save_path)


if __name__ == "__main__":
    main()


__all__ = ["str_to_timestamp", "plot_training_loss", "plot_traing_loss", "main"]
