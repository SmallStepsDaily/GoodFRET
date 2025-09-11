import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Nature 风格
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
})


def generate_boxplot(all_groups_data, output_dir):
    """使用 seaborn 绘制箱型图+散点，保证每个组都显示"""

    # 合并所有组数据到一个 DataFrame
    plot_df = []
    for group, data in all_groups_data.items():
        if not data.empty and 'E' in data.columns and not data['E'].isna().all():
            tmp = data.copy()
            tmp["Group"] = group
            plot_df.append(tmp)
    if not plot_df:
        print("错误：没有有效数据，无法绘制箱型图")
        return

    plot_df = pd.concat(plot_df, ignore_index=True)

    # 确保对照组在最左边
    group_order = ["control"] + [g for g in plot_df["Group"].unique() if g != "control"]

    # 设置绘图风格
    sns.set(style="whitegrid", font="Arial", font_scale=1.2)

    # 绘制箱型图
    plt.figure(figsize=(max(8, 1.5*len(group_order)), 6))
    ax = sns.boxplot(
        x="Group", y="E", data=plot_df, hue="Group", legend=False,
        order=group_order, palette=["#E69F00"] + ["#56B4E9"]*(len(group_order)-1),
        width=0.6, fliersize=0   # 去掉默认异常值显示
    )

    # 添加散点（防止单个点看不见）
    sns.swarmplot(
        x="Group", y="E", data=plot_df,
        order=group_order, color="0.25", size=3, ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("E Value")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "表征值.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"箱型图已保存至: {plot_path}")
    plt.close()

def process_csv(input_file, output_dir='FRET表征值'):
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(input_file)
    print(f"成功读取文件: {input_file}，共 {len(df)} 行数据")

    # 检查必要列
    required_columns = [
        'Metadata_treatment', 'intensity_ratio',
        'Metadata_hour', 'Metadata_concentration', 'ObjectNumber'
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"错误：缺少必要的列 '{col}'")
            return

    # 对照组
    control_group = df[df['Metadata_treatment'] == 'control'].copy()
    if control_group.empty:
        print("错误：未找到对照组数据")
        return
    control_mean = control_group['intensity_ratio'].mean()
    control_std = control_group['intensity_ratio'].std()
    df['E'] = (df['intensity_ratio'] - control_mean) / control_std
    control_group['E'] = (control_group['intensity_ratio'] - control_mean) / control_std

    # 分组
    non_control_df = df[df['Metadata_treatment'] != 'control'].copy()
    non_control_df['Metadata_hour'] = non_control_df['Metadata_hour'].fillna('unknown')
    non_control_df['Metadata_concentration'] = non_control_df['Metadata_concentration'].fillna('unknown')
    non_control_df['group_key'] = (
        non_control_df['Metadata_treatment'] + '_' +
        non_control_df['Metadata_hour'].astype(str) + 'h_' +
        non_control_df['Metadata_concentration'].astype(str) + 'um'
    )
    all_group_keys = non_control_df['group_key'].unique()
    print(f"发现 {len(all_group_keys)} 个非对照组别: {all_group_keys}")

    # 先收集所有组的数据（画图用）
    all_groups_data = {'control': control_group[['E']]}
    for group_key in all_group_keys:
        group_data = non_control_df[non_control_df['group_key'] == group_key]
        all_groups_data[group_key] = group_data[['E']]

    # 先画图
    generate_boxplot(all_groups_data, output_dir)

    # 再输出 CSV（逐组拼接对照组）
    metadata_columns = [col for col in df.columns if col.startswith('Metadata_')]
    columns_to_keep = metadata_columns + ['ObjectNumber', 'E']
    for i, group_key in enumerate(all_group_keys, 1):
        group_data = non_control_df[non_control_df['group_key'] == group_key]
        if group_data.empty:
            print(f"警告：组别 {group_key} 无数据，跳过 CSV 输出")
            continue
        combined_data = pd.concat([
            group_data[columns_to_keep],
            control_group[columns_to_keep]
        ], ignore_index=True)
        output_file = os.path.join(output_dir, f"{group_key}.csv")
        combined_data.to_csv(output_file, index=False)
        print(f"已保存 {i}/{len(all_group_keys)}: {output_file}")

    print("处理完成！")


if __name__ == "__main__":
    input_csv = r"C:\Code\python\csv_data\qrm\20250714\20250709\foxo3a.csv"
    output_dir = r"C:\Code\python\csv_data\qrm\20250714\20250709\Foxo3a表征值"
    process_csv(input_csv, output_dir)
