import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, Tuple
from matplotlib.lines import Line2D
from analysis.fret import FRETCharacterizationValue, save_base64_with_prefix

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 设置图片清晰度

# 自定义配色方案
class CustomPalette:
    CONTROL = "#E0E0E0"  # 浅灰色用于对照组
    TREATMENTS = [
        "#4C72B0", "#55A868", "#C44E52",
        "#8172B2", "#CCB974", "#64B5CD",
        "#DA8BC3", "#8C8C8C", "#CC6677",
        "#332288", "#117733", "#88CCEE"
    ]  # 专业配色方案


class SD(FRETCharacterizationValue):
    def __init__(self, data_df: pd.DataFrame):
        super().__init__(data_df)
        self.result = ''
        self.value = 0
        self.plot = None
        self.time = 0
        self.all_sd_values = {}  # 存储所有SD值结果

    def start(self, control_name: str = 'CTRL', feature_name: str = ''):
        """
        执行SD分析流程，计算各处理组与对照组在不同时间点的SD值
        """
        times = pd.unique(self.data['Metadata_hour'])
        treatments = pd.unique(self.data[self.data['Metadata_treatment'] != control_name]['Metadata_treatment'])
        controls = self.data[self.data['Metadata_treatment'] == control_name]

        # 创建新的DataFrame，保留Metadata_开头的列和ObjectNumber
        metadata_columns = self.metadata_columns
        # 首先计算control在各时间点的分布
        controls_data = {}
        init_control = None
        for time in times:
            time_data = controls[controls['Metadata_hour'] == time]
            controls_data[time] = time_data
            if init_control is None and time_data.empty is False:
                init_control = time_data

        # 标准化表征值
        self.all_sd_values = {}
        sd_drug = {}
        sd_control = {}
        # 计算各处理组在各时间点的分布及SD值
        for time in times:
            data = self.data[self.data['Metadata_hour'] == time]
            control = controls_data[time]  # 使用对应时间点的control分布
            if control.empty:
                control = init_control
            sd_drug[time] = {}
            sd_control[time] = {}
            for treatment in treatments:
                treatment_data = data[data['Metadata_treatment'] == treatment]
                if treatment_data.empty:
                    print(f"警告: 处理组 '{treatment}' 在时间点 {time} 没有有效数据")
                    continue
                # 划分浓度的情况，查看数据下的不同浓度情况
                concentrations = pd.unique(treatment_data['Metadata_concentration'])
                for concentration in concentrations:
                    key_name = f'{treatment}_{time}h_{concentration}um'
                    drug = treatment_data[treatment_data['Metadata_concentration'] == concentration]
                    # 保证drug 和 control都是完整表格便于数据保存
                    E_value, sd_drug[time][f'{treatment}-{concentration}μm'], sd_control[time][f'{treatment}-{concentration}μm'] \
                        = self.compute(control[feature_name], drug[feature_name])
                    # 将获得E值结果进行合并 拼接返回df格式
                    drug_new_df = drug[metadata_columns].copy()
                    drug_new_df['E'] = sd_drug[time][f'{treatment}-{concentration}μm']
                    control_new_df = control[metadata_columns].copy()
                    control_new_df['E'] = sd_control[time][f'{treatment}-{concentration}μm']

                    self.all_sd_values[key_name] = pd.concat([drug_new_df, control_new_df], axis=0)
                    # 更新结果文本
                    self.result += f"{str(time)}h-{control_name} vs {key_name}: SD值 = {E_value:.4f}\n"
        # 生成统计箱型图
        # TODO 缺少浓度的输入情况
        self.plot = self.draw_plt(sd_drug)
        return self.all_sd_values, self.result, self.plot

    def save_dict_to_csv_files(self, save_path):
        """
        将字典中的DataFrame保存为独立的CSV文件

        参数:
        data_dict (dict): 键值对字典，键为文件名，值为pandas DataFrame
        save_path (str): 保存CSV文件的目标文件夹路径

        返回:
        None
        """
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 遍历字典中的每个项目
        for key, df in self.all_sd_values.items():
            try:
                # 构建完整的文件路径
                file_path = os.path.join(save_path, f"{key}.csv")

                # 直接保存DataFrame
                df.to_csv(file_path, index=False)

                print(f"成功保存文件: {file_path}")

            except Exception as e:
                print(f"错误: 保存键 '{key}' 时出错 - {str(e)}")

    def compute(self, control: pd.DataFrame, drug: pd.DataFrame) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        计算单个处理组与对照组的SD值

        返回:
            - SD值
            - 处理组单个细胞的E值
            - 对照组单个细胞的E值
        """
        # 计算去除异常值后的SD值（使用IQR过滤）
        def remove_outliers_by_iqr(data, multiplier=1.5):
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            return data[(data >= lower_bound) & (data <= upper_bound)]
        # 处理存在nan值的情况，采用均值填充
        control_mean_val = control.mean()
        drug_mean_val = drug.mean()
        # 填充 NaN
        control = control.fillna(control_mean_val)
        drug = drug.fillna(drug_mean_val)
        control_std = control.std()
        single_cell_drug_E_value = (drug - control_mean_val) / control_std
        single_cell_control_E_value = (control - control_mean_val) / control_std
        return float(abs(remove_outliers_by_iqr(single_cell_drug_E_value).mean())), single_cell_drug_E_value, single_cell_control_E_value

    def draw_plt(self, sd_drug: Dict[float, Dict[str, pd.DataFrame]]) -> str:
        """
        绘制带人为定义Control基准的E值箱型图（0h固定为Control=0），图例位于图像右侧

        参数:
            sd_drug: 数据字典，结构为 {时间: {处理组: E值数组}}（不含Control数据）

        返回:
            - 图像的base64编码字符串
        """
        # 强制包含0h作为第一个时间点
        times = sorted(sd_drug.keys())

        if 0 not in times:
            times.insert(0, 0)  # 确保0h始终是第一个时间点
        num_times = len(times)

        # 提取非0h的处理组（Control仅在0h人为定义）
        treatments = sorted({treat for t in times if t != 0 for treat in sd_drug.get(t, {}).keys()})
        num_treatments = len(treatments)

        # 设置英文字体
        plt.rcParams["font.family"] = ["Arial", "sans-serif"]

        # 动态计算图表尺寸（增加右侧空间用于图例）
        fig_width = max(10, num_times * 3 + num_treatments * 0.5)  # 增加宽度
        fig_height = 6 if num_treatments <= 3 else 8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # 图表标题
        plt.title('E Value Distribution with Control Baseline', fontsize=16, color='#333', pad=20)
        sns.set_style("whitegrid", {'grid.linestyle': '--'})

        # 可视化参数
        box_width = 0.6 / max(1, num_treatments)  # 非0h处理组宽度
        control_marker = 'x'  # 使用标准x标记
        marker_size = 18
        marker_color = '#dc2624'  # 红色
        marker_edge_width = 3
        baseline_color = '#0072BD'  # 基准线颜色

        # 绘制基准线
        ax.axhline(y=0, color=baseline_color, linestyle='--', alpha=0.7, linewidth=1.5)

        # 存储处理组图例
        legend_handles = []
        legend_labels = treatments.copy()
        legend_flag = {}
        for treatment in treatments:
            legend_flag[treatment] = True

        # 绘制主图表
        for i, time in enumerate(times):
            x_base = i * 2.0  # 扩大时间点间距
            if time == 0:
                # 0h位置绘制Control标记（人为定义E=0）
                ax.plot(x_base, 0,
                        marker=control_marker, markersize=marker_size,
                        markeredgewidth=marker_edge_width, color=marker_color,
                        linestyle='none', label='Control (E=0)')

            else:
                # 非0h时间点绘制处理组箱型图
                time_data = sd_drug[time]

                for j, treat in enumerate(treatments):
                    if time_data.get(treat) is None:
                        continue
                    data = time_data[treat].values
                    x_pos = x_base + j * box_width
                    bp = ax.boxplot(
                        data, positions=[x_pos], widths=box_width,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=sns.color_palette()[j], edgecolor='black'),
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.2),
                        capprops=dict(color='black', linewidth=1.2)
                    )

                    # 为每个处理组收集一次图例（仅需一次）
                    if legend_flag[treat]:
                        legend_handles.append(bp['boxes'][0])
                        legend_flag[treat] = False

        # 设置横坐标
        ax.set_xticks([i * 2.0 for i in range(num_times)])
        ax.set_xticklabels(['0h'] + [f"{t}h" for t in times if t != 0],
                           fontsize=12, rotation=0, ha='center')
        ax.set_ylabel('E Value', fontsize=14, labelpad=15)

        # 创建复合图例（移至图像右侧）
        control_handle = Line2D([0], [0], marker=control_marker, color='white',
                                markerfacecolor=marker_color, markeredgecolor=marker_color,
                                markersize=marker_size, markeredgewidth=marker_edge_width,
                                linestyle='none')

        # 计算图例行数，优化布局
        ncol = 1 if num_treatments <= 5 else 2  # 超过5个处理组时使用2列


        legend = ax.legend(
            [control_handle] + legend_handles,
            ['Control (E=0)'] + legend_labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),  # 定位到图像右侧中心
            frameon=True,
            fancybox=True,
            shadow=True,
            title='Groups',
            fontsize=11,
            title_fontsize=12,
            ncol=ncol,  # 动态设置列数
            borderaxespad=0.5  # 与图像保持距离
        )
        legend.get_title().set_fontweight('bold')

        # 添加范围说明
        plt.figtext(0.01, 0.98, "Note: 'x' = Manually defined Control baseline at 0h",
                    ha="left", fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=3))

        # 优化布局（为右侧图例留出空间）
        plt.tight_layout(pad=5, rect=[0, 0, 0.85, 1])  # 右侧留出15%空间

        # 图像渲染
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f'data:image/png;base64,{image_base64}'



if __name__ == '__main__':
    data_df = pd.read_csv(r"C:\Code\python\csv_data\qrm\20250509\FRET.csv")
    sd_model = SD(data_df)
    # Cell_Ed_region_top_50_value
    values, result_str, image = sd_model.start(feature_name='Ed_region_mean')
    save_base64_with_prefix(image, r"C:\Users\pengs\Downloads\test.png")
    print(result_str)
