import os
from pathlib import Path

def delete_files_by_name(folder_path, target_names):
    """
    删除指定文件夹及其子文件夹中所有指定名称（含扩展名）的文件

    :param folder_path: 文件夹路径（字符串或 Path）
    :param target_names: 要删除的文件名列表（含扩展名），如 ['mask.tif', 'temp.txt']
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"❌ 提供的路径不是一个有效文件夹：{folder}")
        return

    if isinstance(target_names, str):
        target_names = [target_names]

    count = 0
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.name in target_names:
            try:
                file_path.unlink()
                print(f"🗑️ 已删除: {file_path}")
                count += 1
            except Exception as e:
                print(f"⚠️ 删除失败: {file_path}，错误: {e}")

    print(f"✅ 完成，共删除 {count} 个匹配文件。")

if __name__ == '__main__':
    delete_files_by_name(r'D:\data\20250513\BCLXL-BAK', 'seeds_mask.tif')
