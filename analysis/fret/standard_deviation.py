import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional

from analysis.fret import FRETCharacterizationValue

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 设置图片清晰度


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
        for time in times:
            time_data = controls[controls['Metadata_hour'] == time]
            control_feature = time_data[feature_name].values[time_data[feature_name] > 0]
            controls_data[time] = control_feature

        # 标准化表征值
        self.all_sd_values = {}
        sd_drug = {}
        sd_control = {}
        # 计算各处理组在各时间点的分布及SD值
        for time in times:
            data = self.data[self.data['Metadata_hour'] == time]
            control = controls_data[time]  # 使用对应时间点的control分布
            sd_values = {}
            sd_drug[time] = {}
            sd_control[time] = {}
            for treatment in treatments:
                treatment_data = data[data['Metadata_treatment'] == treatment]
                drug = treatment_data[feature_name].values[treatment_data[feature_name] > 0]

                if len(drug) == 0:
                    print(f"警告: 处理组 '{treatment}' 在时间点 {time} 没有有效数据")
                    continue

                # 使用{time}-{treatment}作为键
                key = f"{time}h-{treatment}"
                E_value, sd_drug[time][treatment], sd_control[time][treatment] = self.compute(control, drug)
                sd_values[key] = E_value
                # 更新结果文本
                self.result += f"{str(time)}h-{control_name} vs {key}: SD值 = {E_value:.4f}\n"
            self.all_sd_values[time] = sd_values

        # 生成统计箱型图
        self.plot = self.draw_plt(sd_drug, sd_control, treatments, times, control_name)

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
        绘制不同处理组在各时间点的SD值统计箱型图，以时间为横坐标

        返回:
            - 图像的base64编码字符串
        """
        # 设置图片清晰度
        plt.rcParams['figure.dpi'] = 300

        # 确定箱型图的位置和宽度
        num_treatments = len(treatments)
        num_times = len(times)
        box_width = 0.8 / num_treatments
        spacing = 0.2  # 时间点之间的间距

        # 创建画布
        fig, ax = plt.subplots(figsize=(10 + num_times, 8))

        # 颜色映射
        colors = plt.cm.tab10.colors

        # 用于存储所有箱型图对象，以便创建图例
        boxplot_handles = []
        legend_labels = []

        # 绘制箱型图
        for i, time in enumerate(times):
            position_base = i * (num_treatments * box_width + spacing)

            # 绘制对照组（每个时间点只绘制一次）
            if time in sd_control and control_name in sd_control[time]:
                control_data = sd_control[time][control_name]
                boxprops = dict(linestyle='-', linewidth=1.5, color='black')
                whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')
                medianprops = dict(linestyle='-', linewidth=2.0, color='firebrick')
                flierprops = dict(marker='o', markerfacecolor='green', markersize=8, alpha=0.5)

                bp = ax.boxplot(
                    [control_data],
                    positions=[position_base],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    medianprops=medianprops,
                    flierprops=flierprops
                )

                # 为箱型图填充颜色
                for patch in bp['boxes']:
                    patch.set_facecolor('lightgray')

                # 保存对照组的箱型图对象用于图例
                if not boxplot_handles:
                    boxplot_handles.append(bp['boxes'][0])
                    legend_labels.append(control_name)

            # 绘制各处理组
            for j, treatment in enumerate(treatments):
                position = position_base + (j + 1) * box_width

                if time in sd_drug and treatment in sd_drug[time]:
                    treatment_data = sd_drug[time][treatment]
                    boxprops = dict(linestyle='-', linewidth=1.5)
                    whiskerprops = dict(linestyle='-', linewidth=1.5)
                    medianprops = dict(linestyle='-', linewidth=2.0, color='firebrick')
                    flierprops = dict(marker='o', markerfacecolor='green', markersize=8, alpha=0.5)

                    bp = ax.boxplot(
                        [treatment_data],
                        positions=[position],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        medianprops=medianprops,
                        flierprops=flierprops
                    )

                    # 为箱型图填充颜色
                    color_idx = j % len(colors)
                    for patch in bp['boxes']:
                        patch.set_facecolor(colors[color_idx])

                    # 保存第一个时间点的处理组箱型图对象用于图例
                    if i == 0:
                        boxplot_handles.append(bp['boxes'][0])
                        legend_labels.append(treatment)

        # 设置x轴刻度和标签
        x_ticks = [i * (num_treatments * box_width + spacing) + (num_treatments * box_width) / 2
                   for i in range(num_times)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{time}h" for time in times])

        # 添加图例
        ax.legend(boxplot_handles, legend_labels, loc='upper right')

        # 添加标题和标签
        ax.set_title('不同处理组在各时间点的SD值分布', fontsize=16)
        ax.set_xlabel('时间', fontsize=14)
        ax.set_ylabel('标准化差异值 (SD)', fontsize=14)

        # 添加网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 调整布局
        plt.tight_layout()
        plt.show()
        # 将图像转换为base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        image_base64 = base64.b64encode(image_png).decode('utf-8')
        plt.close()

        return image_base64


if __name__ == '__main__':
    data_df = pd.read_csv(r'D:\data\hql\2025.04.30 fret hoechst mito BF\FRET.csv')
    sd_model = SD(data_df)
    values, result_str, image = sd_model.start(feature_name='Mit_Ed_agg_top_50_value')
    print(result_str)