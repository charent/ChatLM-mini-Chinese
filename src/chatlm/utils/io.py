"""文件与磁盘相关工具。"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import field, make_dataclass

import ujson


def get_free_space_of_disk(folder: str = "./") -> float:
    """获取指定目录所在磁盘剩余空间，返回单位: GB。"""
    res_val = 0.0
    if platform.system() == "Windows":
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(  # type: ignore[attr-defined]
            ctypes.c_wchar_p(folder), None, None, ctypes.pointer(free_bytes)
        )
        res_val = free_bytes.value
    else:
        # statvfs 仅存在于 POSIX 平台，Windows 分支不会走到这里
        st = os.statvfs(folder)  # pyright: ignore[reportAttributeAccessIssue]
        res_val = st.f_bavail * st.f_frsize

    return res_val / (1024**3)


def get_path_of_suffix_files(root: str, suffix: str, with_create_time: bool = False) -> list:
    """获取指定目录下指定后缀的所有文件的绝对路径。"""
    suffix_files = []
    for current_root, _, files in os.walk(root):
        for file in files:
            if file.endswith(suffix):
                full_path = os.path.join(current_root, file)
                if with_create_time:
                    suffix_files.append((full_path, os.path.getctime(full_path)))
                else:
                    suffix_files.append(full_path)

    return suffix_files


def save_model_config(config_dict: dict, file: str) -> None:
    """将模型配置写入到 json 文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        ujson.dump(config_dict, f, indent=4, ensure_ascii=False)


def json_to_dataclass(json_file: str, class_name: str = "Config") -> type:
    """将 json 配置文件转换为 dataclass。

    >>> data_class = json_to_dataclass('my_config.json', 'Config')
    >>> my_config = data_class()
    >>> assert my_config.name == 'Alice'
    """
    with open(json_file, encoding="utf-8") as f:
        json_dict = ujson.load(f)

    # 将 dict 转换为可迭代的属性名称、属性类型、默认值
    fields_list = [(k, type(v), field(default=v)) for k, v in json_dict.items()]

    return make_dataclass(cls_name=class_name, fields=fields_list)


__all__ = [
    "get_free_space_of_disk",
    "get_path_of_suffix_files",
    "save_model_config",
    "json_to_dataclass",
]
