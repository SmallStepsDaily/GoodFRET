"""
通过设定的函数计算对应的PT值
"""


import os

import numpy as np
import pandas as pd

def calculate_A(S, E):
    """
    根据S值和E值计算A值
    这里使用一个示例公式，你可以根据实际需求修改
    """
    # 示例公式：A = S * (1 - E/100)
    E = np.abs(E)
    return 100 * ((0.5 * E + 1) ** (0.5 * S) - 1)


def calculate_B(concentration, hour):
    """
    根据浓度和时间计算B值
    这里使用一个示例公式，你可以根据实际需求修改
    """
    # 示例公式：B = concentration * hour / 100
    hour = hour / 24
    # 浓度单位默认时 μm 为单位，乘上10-6进行转化
    concentration = concentration * 1e-6
    B_matrix = (-np.log10(concentration + 1e-10)) ** (-hour)
    return B_matrix / (1 + B_matrix)


def calculate_PT(A, B):
    """
    根据A值和B值计算PT值
    这里使用一个示例公式，你可以根据实际需求修改
    """
    # 示例公式：PT = A * B^0.5
    return A * B


def process_single_cell_files(single_cell_dir):
    """
    处理单细胞表征值文件夹中的所有CSV文件
    计算A、B、PT值并添加到原文件中
    """
    if not os.path.exists(single_cell_dir):
        print(f"错误: 单细胞表征值文件夹 '{single_cell_dir}' 不存在")
        return

    # 遍历所有CSV文件
    for filename in os.listdir(single_cell_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(single_cell_dir, filename)
            try:
                df = pd.read_csv(file_path)

                # 检查必要的列是否存在
                required_cols = ['S', 'E', 'Metadata_concentration', 'Metadata_hour']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    print(f"警告: 文件 {filename} 缺少必要的列 {missing_cols}，无法计算A、B、PT值，已跳过")
                    continue

                # 计算新特征
                df['A'] = calculate_A(df['S'], df['E'])
                df['B'] = calculate_B(df['Metadata_concentration'], df['Metadata_hour'])
                df['PT'] = calculate_PT(df['A'], df['B'])

                # 保存更新后的文件
                df.to_csv(file_path, index=False)
                print(f"已更新文件 {filename}，添加了A、B、PT值")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


def main(output_dir):

    print("处理单细胞表征值文件，计算A、B、PT值...")
    process_single_cell_files(output_dir)
    print("所有处理完成！")


if __name__ == "__main__":
    # 示例用法
    output_directory = r'C:\Code\python\csv_data\qrm\20250714\20250705\单细胞表征值'  # 输出文件夹路径
    main(output_directory)
