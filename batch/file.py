import os
import re
from pathlib import Path


def list_immediate_subdirectories(path):
    """
    列出指定路径下一级的所有子文件夹名称。

    :param path: 要遍历的根目录路径
    :return: 子文件夹名称的列表
    """
    try:
        return [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]
    except FileNotFoundError:
        print(f"指定的路径 {path} 不存在。")
        return []


def list_numeric_subdirectories(path):
    """
    列出指定路径下一级的所有名称为数字的子文件夹。

    :param path: 要遍历的根目录路径
    :return: 数字名称子文件夹的列表
    """
    numeric_pattern = re.compile(r'^\d+$')  # 匹配纯数字的正则表达式

    try:
        return [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name)) and numeric_pattern.match(name)]
    except FileNotFoundError:
        print(f"指定的路径 {path} 不存在。")
        return []
