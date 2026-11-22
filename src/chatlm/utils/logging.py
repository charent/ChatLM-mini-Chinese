"""彩色控制台 + 文件双通道日志。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import colorlog

from chatlm.paths import LOG_DIR


class Logger:
    """自定义日志格式。

    参数:
        logger_name: logger 名称前缀
        level: 日志级别
        std_out: 是否输出到控制台
        save2file: 是否写入日志文件
        file_name: 指定的日志文件路径，默认为 ``<artifacts>/logs/<name>-<date>.log``
    """

    def __init__(
        self,
        logger_name: str,
        level: int = logging.DEBUG,
        std_out: bool = True,
        save2file: bool = False,
        file_name: str | None = None,
    ) -> None:
        if std_out is False and save2file is False:
            raise ValueError("args: [std_out, save2file], at less one of them must be True")

        datefmt = "%Y-%m-%d %H:%M:%S"

        # 输出到控制台
        if std_out:
            std_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(log_color)s%(message)s"

            self.stdout_logger = logging.getLogger(f"{logger_name}_std")
            self.stdout_logger.setLevel(level)

            # 彩色输出格式化
            log_colors_config = {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            }
            formatter = colorlog.ColoredFormatter(
                fmt=std_logfmt,
                datefmt=datefmt,
                log_colors=log_colors_config,
            )

            # 避免同一 logger 被重复初始化时重复挂载 handler
            if not self.stdout_logger.handlers:
                sh = logging.StreamHandler()
                sh.setLevel(level)
                sh.setFormatter(formatter)
                self.stdout_logger.addHandler(sh)

        # 输出到文件
        if save2file:
            file_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s"

            self.file_logger = logging.getLogger(f"{logger_name}_file")
            self.file_logger.setLevel(level)

            if file_name is not None:
                log_file = Path(file_name)
            else:
                log_file = LOG_DIR / f"{logger_name}-{time.strftime('%Y%m%d', time.localtime())}.log"

            log_file.parent.mkdir(parents=True, exist_ok=True)

            if not self.file_logger.handlers:
                fh = logging.FileHandler(filename=str(log_file), mode="a", encoding="utf-8")
                fh.setLevel(level)
                save_formatter = logging.Formatter(fmt=file_logfmt, datefmt=datefmt)
                fh.setFormatter(save_formatter)
                self.file_logger.addHandler(fh)

    def info(self, message: str, std_out: bool = True, save_to_file: bool = False) -> None:
        if std_out:
            self.stdout_logger.info(message)
        if save_to_file:
            self.file_logger.info(message)

    def debug(self, message: str, std_out: bool = True, save_to_file: bool = False) -> None:
        if std_out:
            self.stdout_logger.debug(message)
        if save_to_file:
            self.file_logger.debug(message)

    def warning(self, message: str, std_out: bool = True, save_to_file: bool = False) -> None:
        if std_out:
            self.stdout_logger.warning(message)
        if save_to_file:
            self.file_logger.warning(message)

    def error(self, message: str, std_out: bool = True, save_to_file: bool = False) -> None:
        if std_out:
            self.stdout_logger.error(message)
        if save_to_file:
            self.file_logger.error(message)


__all__ = ["Logger"]
