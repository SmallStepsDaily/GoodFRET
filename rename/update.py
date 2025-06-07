import os
from pathlib import Path

from ui import Output


def have_target_image(dir_path, target_image_names):
    """
    查看指定目录中是否包含所有目标文件，且文件名大小写敏感。

    :param dir_path: 要检查的目录路径（字符串或 Path 对象）
    :param target_image_names: 目标文件名列表（字符串列表）
    :return: 如果所有目标文件都存在返回 True，否则返回 False
    """
    dir_path = Path(dir_path)

    # 确保目录存在且是一个有效的目录
    if not dir_path.is_dir():
        return False

    # 获取目录下所有文件的名字（不包括子目录）
    existing_files = {entry.name for entry in dir_path.iterdir() if entry.is_file()}

    # 检查所有目标文件是否存在于现有文件集合中，并且大小写敏感
    return all(name in existing_files for name in target_image_names)


def update_file_name(process_file_path, have_files_name, chang_name, output=Output()):
    """
    修改文件名称
    process_file_path: 文件路径
    have_files_name: 已经存在的文件名称列表
    chang_name: 需要更改的文件名称列表
    """
    # 查看需要的已有图像是否存在 不存在报错
    if not have_target_image(process_file_path, have_files_name):
        print(f"指定图像缺失++++++++++++++++++++++++{process_file_path}")
        output.append(f"指定图像缺失++++++++++++++++++++++++{process_file_path}")
        return
    # 检验是否存在目标图像
    if have_target_image(process_file_path, chang_name):
        print(f"该图像集内存在所指定的图像 : {process_file_path}")
        output.append(f"该图像集内存在所指定的图像 : {process_file_path}")
        return

    # 过滤出以 'image_' 开头的 .tif 文件
    image_files = [f for f in os.listdir(process_file_path) if f.startswith('image_') and f.lower().endswith('.tif')]
    name_list_length = len(chang_name)
    if len(image_files) != name_list_length:
        print("该图像集存在问题++++++++++++++++++++++++", process_file_path)
        output.append(f"该图像集存在问题++++++++++++++++++++++++{process_file_path}")
        return

    # 确保按照文件名排序，以便正确识别第一张和第二张图像
    image_files.sort()

    try:
        # 获取文件的完整路径
        image_paths = [Path(process_file_path) / img for img in image_files]
        for i in range(0, name_list_length):
            # 重命名图像为 mit.tif
            image_paths[i].rename(Path(process_file_path) / chang_name[i])
        print(f"成功处理文件夹 : {process_file_path}")
        output.append(f"成功处理文件夹 : {process_file_path}")
    except Exception as e:
        print(f"处理文件夹 {process_file_path} 时发生错误: {e}")
        output.append(f"处理文件夹 {process_file_path} 时发生错误: {e}")