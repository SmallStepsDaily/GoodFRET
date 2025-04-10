import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot_by_treatment(df, treatment_column='Metadata_treatment', ratio_column='intensity_ratio', control_label='control'):
    # 确保 control 组在最左边
    treatments = df[treatment_column].unique().tolist()
    if control_label in treatments:
        treatments.remove(control_label)
        treatments = [control_label] + treatments

    # 设置图形样式
    sns.set(style="whitegrid")

    # 定义一组配色方案
    palette = sns.color_palette("husl", len(treatments))

    # 绘制箱型图，修改参数以避免警告
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=treatment_column, y=ratio_column, data=df, order=treatments, showfliers=False, hue=treatment_column, palette=palette, legend=False)

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
    plt.ylabel('Foxo3a Intensity Ratio')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    path = r"D:\data\qrm\2025.03.19 PC9 FOXO3A 4H\foxo3a.csv"
    df = pd.read_csv(path)
    plot_boxplot_by_treatment(df)