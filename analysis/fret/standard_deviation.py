import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional

from sympy.stats.sampling.sample_numpy import numpy

from analysis.fret import FRETCharacterizationValue

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
        self.data = data_df
        self.result = ''
        self.value = 0
        self.plot = None
        self.time = 0
        self.all_sd_values = {}  # 存储所有SD值结果

    def start(self, control_name: str = 'control', feature_name: str = ''):
        """
        执行SD分析流程，计算各处理组与对照组在不同时间点的SD值
        """
        times = pd.unique(self.data['Metadata_hour'])
        treatments = pd.unique(self.data[self.data['Metadata_treatment'] != control_name]['Metadata_treatment'])
        controls = self.data[self.data['Metadata_treatment'] == control_name]

        # 首先计算control在各时间点的分布
        controls_data = {}
        init_control = None
        for time in times:
            time_data = controls[controls['Metadata_hour'] == time]
            control_feature = time_data[feature_name].values[time_data[feature_name] > 0]
            controls_data[time] = control_feature
            if init_control is None and len(control_feature) > 0:
                init_control = control_feature

        # 标准化表征值
        self.all_sd_values = {}
        sd_drug = {}
        sd_control = {}
        treatment_list = []
        # 计算各处理组在各时间点的分布及SD值
        for time in times:
            data = self.data[self.data['Metadata_hour'] == time]
            control = controls_data[time]  # 使用对应时间点的control分布
            if len(control) == 0:
                control = init_control
            sd_values = {}
            sd_drug[time] = {}
            sd_control[time] = {}
            for treatment in treatments:
                treatment_data = data[data['Metadata_treatment'] == treatment]
                if len(treatment_data) == 0:
                    print(f"警告: 处理组 '{treatment}' 在时间点 {time} 没有有效数据")
                    continue
                # 划分浓度的情况，查看数据下的不同浓度情况
                concentrations = pd.unique(treatment_data['Metadata_concentration'])
                if len(concentrations) > 1:
                    sd_drug[time][treatment] = {}
                    sd_control[time][treatment] = {}
                    for concentration in concentrations:
                        key = f"{time}h-{treatment}-{concentration}μm"
                        drug = treatment_data[treatment_data['Metadata_concentration'] == concentration][feature_name].values
                        E_value, sd_drug[time][f'{treatment}-{concentration}μm'], sd_control[time][f'{treatment}-{concentration}μm'] = self.compute(control, drug)
                        treatment_list.append(f'{treatment}-{concentration}μm')
                        sd_values[key] = E_value
                        # 更新结果文本
                        self.result += f"{str(time)}h-{control_name} vs {key}: SD值 = {E_value:.4f}\n"
                # 如果没有浓度梯度的情况进行统计
                else:
                    drug = treatment_data[feature_name].values[treatment_data[feature_name] > 0]
                    # 使用{time}-{treatment}作为键
                    key = f"{time}h-{treatment}"
                    E_value, sd_drug[time][treatment], sd_control[time][treatment] = self.compute(control, drug)
                    treatment_list.append(treatment)
                    sd_values[key] = E_value
                    # 更新结果文本
                    self.result += f"{str(time)}h-{control_name} vs {key}: SD值 = {E_value:.4f}\n"
            self.all_sd_values[time] = sd_values

        # 生成统计箱型图
        # TODO 缺少浓度的输入情况
        self.plot = self.draw_plt(sd_drug, sd_control, treatment_list, times, control_name)

        return self.all_sd_values, self.result, self.plot

    def compute(self, control: np.ndarray, drug: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        计算单个处理组与对照组的SD值

        返回:
            - SD值
            - 处理组单个细胞的E值
            - 对照组单个细胞的E值
        """
        control_mean = control.mean()
        control_std = control.std()
        single_cell_drug_E_value = (drug - control_mean) / control_std
        single_cell_control_E_value = (control - control_mean) / control_std
        return abs(single_cell_drug_E_value.mean()), single_cell_drug_E_value, single_cell_control_E_value

    def draw_plt(self, sd_drug: Dict[float, Dict[str, np.ndarray]],
                 sd_control: Dict[float, Dict[str, np.ndarray]],
                 treatments: List[str],
                 times: List[float],
                 control_name: str) -> str:
        """
        绘制增强版的不同处理组在各时间点的SD值统计箱型图

        返回:
            - 图像的base64编码字符串
        """
        # 优化图表尺寸计算，根据时间点数量动态调整宽度
        num_treatments = len(treatments)
        num_times = len(times)

        # 动态计算图表宽度，每个时间点大约占3英寸
        fig_width = max(6, num_times * 2.5)
        fig_height = 8

        # 创建画布
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        plt.title('标准化差异值分析', fontsize=14, color='#666', pad=10)

        # 确定箱型图的位置和宽度
        box_width = 0.8 / (num_treatments + 1)  # 调整宽度计算
        spacing = 0.3  # 时间点之间的间距

        # 颜色映射
        colors = CustomPalette.TREATMENTS

        # 用于存储所有箱型图对象，以便创建图例
        boxplot_handles = []
        legend_labels = []

        # 绘制箱型图
        for i, time in enumerate(times):
            position_base = i * ((num_treatments + 1) * box_width + spacing)

            # 绘制对照组（每个时间点只绘制一次）
            if time in sd_control and control_name in sd_control[time]:
                control_data = sd_control[time][control_name]
                if len(control_data) == 0:
                    break
                # 过滤超出[-2, 2]范围的数据点
                filtered_control = np.clip(control_data, -2, 2)

                # 绘制带有增强视觉效果的箱型图（不显示异常值）
                boxprops = dict(linestyle='-', linewidth=2, color='#333')
                whiskerprops = dict(linestyle='-', linewidth=1.5, color='#333')
                medianprops = dict(linestyle='-', linewidth=3, color='#D55E00')
                flierprops = dict(marker='None')  # 不显示异常值
                meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=8)

                bp = ax.boxplot(
                    [filtered_control],
                    positions=[position_base],
                    widths=box_width * 1.2,  # 对照组箱型图略宽
                    patch_artist=True,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    medianprops=medianprops,
                    flierprops=flierprops,
                    meanprops=meanprops,
                    showmeans=True
                )

                # 为箱型图填充颜色，添加渐变效果
                for patch in bp['boxes']:
                    patch.set_facecolor(CustomPalette.CONTROL)
                    patch.set_alpha(0.8)

                # 添加轻微的阴影效果
                for patch in bp['boxes']:
                    x, y = patch.get_xy()
                    width, height = patch.get_width(), patch.get_height()
                    shadow = plt.Rectangle((x + 2, y - 2), width, height, fill=True, color='#000', alpha=0.1, zorder=0)
                    ax.add_patch(shadow)

                # 保存对照组的箱型图对象用于图例
                if not boxplot_handles:
                    boxplot_handles.append(bp['boxes'][0])
                    legend_labels.append(control_name)

            # 绘制各处理组
            for j, treatment in enumerate(treatments):
                position = position_base + (j + 1) * box_width

                if time in sd_drug and treatment in sd_drug[time]:
                    treatment_data = sd_drug[time][treatment]

                    # 过滤超出[-2, 2]范围的数据点
                    filtered_treatment = np.clip(treatment_data, -2, 2)

                    # 绘制带有增强视觉效果的箱型图（不显示异常值）
                    boxprops = dict(linestyle='-', linewidth=2)
                    whiskerprops = dict(linestyle='-', linewidth=1.5)
                    medianprops = dict(linestyle='-', linewidth=3, color='#D55E00')
                    flierprops = dict(marker='None')  # 不显示异常值
                    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=8)

                    bp = ax.boxplot(
                        [filtered_treatment],
                        positions=[position],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        medianprops=medianprops,
                        flierprops=flierprops,
                        meanprops=meanprops,
                        showmeans=True
                    )

                    # 为箱型图填充颜色，添加渐变效果
                    color_idx = j % len(colors)
                    for patch in bp['boxes']:
                        patch.set_facecolor(colors[color_idx])
                        patch.set_alpha(0.8)

                    # 保存第一个时间点的处理组箱型图对象用于图例
                    if i == 0:
                        boxplot_handles.append(bp['boxes'][0])
                        legend_labels.append(treatment)

        # 设置x轴刻度和标签
        x_ticks = [i * ((num_treatments + 1) * box_width + spacing) + ((num_treatments + 1) * box_width) / 2
                   for i in range(num_times)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{time}小时" for time in times], fontsize=12)

        # 添加图例
        legend = ax.legend(boxplot_handles, legend_labels, loc='upper right',
                           frameon=True, fancybox=True, shadow=True,
                           title="处理组", fontsize=11)
        legend.get_title().set_fontsize(12)
        legend.get_title().set_fontweight('bold')

        # 添加标题和标签
        ax.set_xlabel('时间', fontsize=14, labelpad=10)
        ax.set_ylabel('标准化差异值 (SD)', fontsize=14, labelpad=10)

        # 固定y轴范围为-2到2
        ax.set_ylim(-2, 2)

        # 添加网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 添加参考线
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, linewidth=1.5)
        ax.axhline(y=-2, color='g', linestyle=':', alpha=0.3, linewidth=1)
        ax.axhline(y=2, color='g', linestyle=':', alpha=0.3, linewidth=1)

        # 美化边框
        for spine in ax.spines.values():
            spine.set_color('#ccc')

        # 添加数据来源注释
        plt.figtext(0.99, 0.01, f"数据来源: 共{len(treatments)}个处理组，{num_times}个时间点",
                    ha="right", fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        # 添加范围说明
        plt.figtext(0.01, 0.01, "注: 超出[-2, 2]范围的数据已被截断",
                    ha="left", fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 3})

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 为标题留出空间

        # 将图像转换为base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        image_base64 = base64.b64encode(image_png).decode('utf-8')
        plt.close()

        return f'data:image/png;base64,{image_base64}'


if __name__ == '__main__':
    data_df = pd.read_csv(r"C:\Code\python\csv_data\gl\20250509\20250412-20250513对照组FRET.csv")
    sd_model = SD(data_df)
    values, result_str, image = sd_model.start(feature_name='Cell_Ed_agg_top_50_value')
    print(result_str)