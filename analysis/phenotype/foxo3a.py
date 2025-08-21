import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_boxplot_by_treatment(df, treatment_column='Metadata_treatment', ratio_column='intensity_ratio',
                              control_label='control', output_dir=None):
    # 确保 control 组在最左边
    treatments = df[treatment_column].unique().tolist()
    if control_label in treatments:
        treatments.remove(control_label)
        treatments = [control_label] + treatments

    df = df.dropna(axis=0)
    # 对指定列求倒数
    df.loc[:, ratio_column] = 1 / df[ratio_column]

    # 计算各处理组的统计数据（均值、中位数、标准差等）
    stats_df = df.groupby(treatment_column)[ratio_column].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()

    # 保存统计数据为CSV
    if output_dir:
        stats_path = os.path.join(output_dir, 'Foxo3a_statistics.csv')
        stats_df.to_csv(stats_path, index=False)
        print(f"统计数据已保存至: {stats_path}")
        print("\n各处理组的统计数据:")
        print(stats_df.to_string(index=False))  # 打印整齐的表格格式

    # 设置图形样式
    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", len(treatments))

    # 绘制箱型图
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=treatment_column, y=ratio_column, data=df, order=treatments, showfliers=False,
                     hue=treatment_column, palette=palette, legend=False)

    # 绘制离群值散点
    for i, treatment in enumerate(treatments):
        treatment_data = df[df[treatment_column] == treatment][ratio_column]
        q1 = treatment_data.quantile(0.25)
        q3 = treatment_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = treatment_data[(treatment_data < lower_bound) | (treatment_data > upper_bound)]
        if not outliers.empty:
            plt.scatter([i] * len(outliers), outliers, color=palette[i], s=20, alpha=0.3)

    # 设置图形标题和坐标轴标签
    plt.title(f'Boxplot of {ratio_column} by {treatment_column}')
    plt.xlabel('Treatment')
    plt.ylabel('Foxo3a Nuclear/Non-Nuclear Intensity Ratio')

    # 保存图形
    if output_dir:
        plot_path = os.path.join(output_dir, 'Foxo3a_boxplot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"箱线图已保存至: {plot_path}")

    # 显示图形
    # plt.show()


if __name__ == '__main__':
    input_path = r"C:\Code\python\csv_data\qrm\20250714\20250709\foxo3a.csv"
    df = pd.read_csv(input_path)
    output_directory = os.path.dirname(input_path)

    plot_boxplot_by_treatment(df, output_dir=output_directory)