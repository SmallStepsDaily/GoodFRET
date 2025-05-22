import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from analysis.fret import FRETCharacterizationValue

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class JSDivergence(FRETCharacterizationValue):
    def __init__(self, data_df):
        self.data = data_df
        self.result = ''
        self.value = 0
        self.plot = None
        self.time = 0

    def start(self, control_name='control', feature_name=''):
        """
        开始分析所有时间点下各处理组与对照组的JS散度，并生成合并图像
        """
        print("开始运行JS散度计算")
        if not feature_name:
            raise ValueError("请提供要分析的特征列名")

        times = pd.unique(self.data['Metadata_hour'])
        treatments = pd.unique(self.data[self.data['Metadata_treatment'] != control_name]['Metadata_treatment'])
        control_data = self.data[self.data['Metadata_treatment'] == control_name]

        all_js_divergence = {}  # 存储各时间点下各处理组的JS散度值

        # 定义x轴范围，基于所有数据
        x_range = self._compute_x_range(feature_name)

        # 存储所有时间点的control和treatment的PDF
        all_pdfs = {}

        # 首先计算control在各时间点的分布
        control_pdfs = {}
        init_control = None
        for time in times:
            # 关键修改：先筛选对应时间点的数据
            time_data = control_data[control_data['Metadata_hour'] == time]
            if len(time_data) == 0:
                time_data = init_control
            else:
                if init_control is None:
                    init_control = time_data
            control_feature = time_data[feature_name].values[time_data[feature_name] > 0]
            pdf_control = self._compute_pdf(control_feature, x_range)
            control_pdfs[time] = pdf_control

        # 计算各处理组在各时间点的分布及JS散度
        for time in times:
            data = self.data[self.data['Metadata_hour'] == time]
            pdf_control = control_pdfs[time]  # 使用对应时间点的control分布

            js_values = {}

            for treatment in treatments:
                treatment_data = data[data['Metadata_treatment'] == treatment]
                treatment_feature = treatment_data[feature_name].values[treatment_data[feature_name] > 0]

                if len(treatment_feature) == 0:
                    print(f"警告: 处理组 '{treatment}' 在时间点 {time} 没有有效数据")
                    continue

                pdf_treatment = self._compute_pdf(treatment_feature, x_range)
                js_divergence_value = jensenshannon(pdf_control, pdf_treatment, base=2)

                # 使用{time}-{treatment}作为键
                key = f"{time}h-{treatment}"
                all_pdfs[key] = pdf_treatment
                js_values[key] = js_divergence_value

                # 更新结果文本
                self.result += f"{control_name} vs {key}: JS散度 = {js_divergence_value:.4f}\n"

            all_js_divergence[time] = js_values

        # 生成合并图像
        plot = self.draw_plt(
            x_range, control_pdfs, all_pdfs, all_js_divergence,
            control_name, times, treatments
        )

        self.value = all_js_divergence
        self.plot = plot
        return self.value, self.result, self.plot

    def _compute_x_range(self, feature_name):
        """计算所有数据的x轴范围"""
        all_data = self.data[feature_name].values[self.data[feature_name] > 0]
        return np.linspace(min(all_data), max(all_data), 1000)

    def _compute_pdf(self, data, x_range):
        """计算数据的概率密度函数"""
        if len(data) < 2:  # 至少需要两个数据点
            return np.zeros_like(x_range)
        kde = gaussian_kde(data)
        pdf = kde(x_range)
        return pdf / pdf.sum()  # 归一化

    def draw_plt(self, x_range, control_pdfs, all_pdfs, all_js_divergence, control_name, times, treatments):
        """绘制合并的图像，包含所有时间点的control和treatment的分布曲线"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from io import BytesIO
        import base64

        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei"]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

        # 为control定义统一的颜色（黑色），不同时间点使用不同线型
        control_colors = ['black']
        control_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # 定义多种线型，用于区分不同时间点的control

        # 为treatment定义颜色循环
        treatment_colors = plt.cm.tab20.colors  # 使用更多颜色

        # 绘制所有时间点的control曲线
        for i, time in enumerate(times):
            pdf_control = control_pdfs[time]
            color = control_colors[0]
            linestyle = control_linestyles[i % len(control_linestyles)]
            ax.plot(x_range, pdf_control, label=f"{time}h-{control_name}",
                    color=color, linestyle=linestyle, linewidth=2)

        print(all_js_divergence)
        # 绘制所有treatment曲线
        for i, (key, pdf) in enumerate(all_pdfs.items()):
            time, treatment = key.split('h-')
            # 计算该treatment在所有处理组中的索引位置
            treatment_index = list(treatments).index(treatment)
            # 计算颜色索引（确保同一treatment在不同时间点使用相同颜色）
            color_index = treatment_index % len(treatment_colors)
            color = treatment_colors[color_index]
            # 获取该treatment的JS散度值
            js_value = all_js_divergence[float(time)][key]

            # 绘制曲线
            ax.plot(x_range, pdf, label=f"{key} (JS={js_value:.4f})",
                    color=color, alpha=0.7)

        # 设置图表标题和标签
        ax.set_title(f"所有时间点的FRET特征分布比较", fontsize=16)
        ax.set_xlabel('特征值', fontsize=14)
        ax.set_ylabel('概率密度', fontsize=14)

        # 隐藏y轴刻度
        ax.set_yticks([])

        # 添加图例
        ax.legend(loc='upper right', fontsize=10, ncol=2)

        # 在底部添加所有JS散度信息
        js_text = []
        for time, js_values in all_js_divergence.items():
            js_text.append(f"时间点 {time}:")
            for key, js in js_values.items():
                js_text.append(f"  {control_name} vs {key}: JS = {js:.4f}")

        fig.text(0.05, 0.01, "\n".join(js_text), ha='left', fontsize=8)

        plt.tight_layout()
        # 保存图像到内存
        canvas = FigureCanvasAgg(fig)
        buffer = BytesIO()
        canvas.print_png(buffer)
        buffer.seek(0)

        image_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        return f'data:image/png;base64,{image_data}'

if __name__ == '__main__':
    data_df = pd.read_csv(r'C:\Code\python\csv_data\gl\20250412\FRET.csv')
    jd_model = JSDivergence(data_df)
    values, result_str, image = jd_model.start(feature_name='Cell_Ed_agg_top_50_value')
    print(result_str)
