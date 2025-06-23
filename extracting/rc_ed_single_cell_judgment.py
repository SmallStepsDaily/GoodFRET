"""
单细胞FRET变化判断函数
1. 使用control组拟合函数
2. 使用加药组组拟合函数
3. 单细胞尺度利用区域效率查看符合哪个拟合函数判断是否发生了 FRET 变化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 定义控制组和药物组的响应函数
def control_ed(rc, edmax=0.3680, kd=1.0):
    return edmax * rc / (kd + rc)


def drug_ed(rc, edmax=0.1819, kd=0.6653):
    return edmax * rc / (kd + rc)


def analyze_fret_data(input_file, output_file=None, save_plot=True, plot_file=None):
    """
    分析 FRET 数据，判断每个数据点更接近 control_ed 还是 drug_ed 函数

    参数:
    input_file (str): 输入 CSV 文件路径
    output_file (str, optional): 输出 CSV 文件路径，默认为 None（在输入文件名后添加 _analyzed）
    save_plot (bool, optional): 是否保存拟合结果图，默认为 True
    plot_file (str, optional): 图像保存路径，默认为 None（在输入文件名后添加 _plot.png）
    """
    # 记录输入文件名用于自动生成图像名
    analyze_fret_data.input_file = input_file

    # 读取 CSV 文件
    try:
        df = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"数据包含 {len(df)} 行和 {len(df.columns)} 列")
    except FileNotFoundError:
        print(f"错误: 文件 {input_file} 不存在")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生错误 - {e}")
        return

    # 提取元数据列 (Metadata_ 开头) 和 ObjectNumber
    metadata_columns = [col for col in df.columns if col.startswith('Metadata_') or col == 'ObjectNumber']

    # 检查所需的特征列是否存在
    required_features = ['Ed_region_mean_value', 'Rc_region_mean']
    for feature in required_features:
        if feature not in df.columns:
            print(f"错误: 数据中缺少必要的列 '{feature}'")
            return

    # 提取数据
    ed_values = df['Ed_region_mean_value']
    rc_values = df['Rc_region_mean']

    # 计算每个点到两个函数的距离
    control_pred = control_ed(rc_values)
    drug_pred = drug_ed(rc_values)

    # 计算绝对误差
    control_error = np.abs(ed_values - control_pred)
    drug_error = np.abs(ed_values - drug_pred)

    # 判断更接近哪个函数
    df['FRET_Judge'] = drug_error < control_error  # True 表示更接近 drug_ed

    # 计算更接近的函数的预测值
    df['Near_Ed'] = np.where(df['FRET_Judge'], drug_pred, control_pred)

    # 准备输出数据，包含元数据、原始特征和新计算的特征
    output_columns = metadata_columns + required_features + ['FRET_Judge', 'Near_Ed']
    output_df = df[output_columns]

    # 确定输出文件名
    if output_file is None:
        output_path = os.path.dirname(input_file)
        output_file = f"{output_path}/Rc-Ed_FRET_analyzed.csv"

    # 保存结果
    try:
        output_df.to_csv(output_file, index=False)
        print(f"分析结果已保存至: {output_file}")

        # 统计结果
        total = len(df)
        drug_count = df['FRET_Judge'].sum()
        control_count = total - drug_count

        print(f"\n统计结果:")
        print(f"总数据点: {total}")
        print(f"更接近 drug_ed 函数的点: {drug_count} ({drug_count / total * 100:.2f}%)")
        print(f"更接近 control_ed 函数的点: {control_count} ({control_count / total * 100:.2f}%)")

    except Exception as e:
        print(f"错误: 保存结果时发生错误 - {e}")

    # 可视化结果
    if save_plot:
        plot_fret_analysis(rc_values, ed_values, control_pred, drug_pred, df['FRET_Judge'], plot_file)


def plot_fret_analysis(rc_values, ed_values, control_pred, drug_pred, judge_results, plot_file=None):
    """
    可视化 FRET 分析结果（只显示非control的数据点）

    参数:
    rc_values: Rc_region_mean 值
    ed_values: Ed_region_mean_value 值
    control_pred: control_ed 函数的预测值
    drug_pred: drug_ed 函数的预测值
    judge_results: FRET_Judge 判断结果
    plot_file: 图像保存路径
    """
    plt.figure(figsize=(10, 6))

    # 只绘制非control的数据点（judge_results为True）
    plt.scatter(rc_values[judge_results], ed_values[judge_results],
                c='red', marker='o', s=30, alpha=0.6, label='Near drug_ed')

    # 绘制两个函数曲线
    rc_range = np.linspace(min(rc_values), max(rc_values), 500)
    plt.plot(rc_range, control_ed(rc_range), 'b-', linewidth=2, label='control_ed')
    plt.plot(rc_range, drug_ed(rc_range), 'r-', linewidth=2, label='drug_ed')

    # 添加图例和标签
    plt.legend(loc='best')
    plt.xlabel('Rc')
    plt.ylabel('Ed')
    plt.title('FRET Data Analysis (Non-control Points)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 保存图像
    if plot_file:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {plot_file}")
    else:
        # 自动生成图像文件名
        if hasattr(analyze_fret_data, 'input_file'):
            output_path = os.path.dirname(analyze_fret_data.input_file)
            output_file = f"{output_path}/Rc-Ed_drug_classification.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {output_file}")

    plt.close()  # 关闭图像窗口，避免重复显示


if __name__ == "__main__":
    # 示例用法
    input_csv = r"C:\Code\python\csv_data\gl\20250513\BCLXL-BAK\FRET.csv"  # 替换为你的 CSV 文件路径
    analyze_fret_data(input_csv, save_plot=True)