import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from io import BytesIO
import base64

from scipy.stats import gaussian_kde

from analysis.fret import FRETCharacterizationValue, save_base64_with_prefix

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 300

class AnomalyScore(FRETCharacterizationValue):
    def __init__(self, data_df: pd.DataFrame):
        super().__init__(data_df)
        self.result = ''
        self.anomaly_scores = {}  # 存储所有计算结果
        self.plot = None

    def start(self, control_name: str = 'control', feature_name: str = '') -> Tuple[Dict[str, pd.DataFrame], str, str]:
        """
        主流程：
        - 根据时间点分组，计算控制组的CDF
        - 计算各处理组相对于控制组的异常分数（双尾）
        - 生成统计文本和箱型图Base64字符串

        返回：
            - 各组异常分数数据字典 {时间_组名_浓度: DataFrame}
            - 统计结果字符串
            - 图像base64字符串
        """
        times = pd.unique(self.data['Metadata_hour'])
        treatments = pd.unique(self.data[self.data['Metadata_treatment'] != control_name]['Metadata_treatment'])
        controls = self.data[self.data['Metadata_treatment'] == control_name]

        metadata_columns = self.metadata_columns

        self.anomaly_scores = {}
        result_lines = []
        controls_data = {}
        init_control = None
        for time in times:
            time_data = controls[controls['Metadata_hour'] == time]
            controls_data[time] = time_data
            if not time_data.empty:
                if init_control is None:
                    init_control = time_data

        for time in times:
            data_time = self.data[self.data['Metadata_hour'] == time]
            control = controls_data.get(time, pd.DataFrame())
            if control.empty:
                control = init_control
            for treatment in treatments:
                treatment_data = data_time[data_time['Metadata_treatment'] == treatment]
                if treatment_data.empty:
                    continue
                concentrations = pd.unique(treatment_data['Metadata_concentration'])
                for conc in concentrations:
                    key_name = f"{treatment}_{time}h_{conc}um"
                    drug = treatment_data[treatment_data['Metadata_concentration'] == conc]
                    # 计算异常分数
                    mean_score, drug_scores, control_scores = self.compute(control[feature_name], drug[feature_name])
                    # 合并数据方便后续保存或分析
                    drug_df = drug[metadata_columns].copy()
                    drug_df['E'] = drug_scores
                    control_df = control[metadata_columns].copy()
                    control_df['E'] = control_scores
                    self.anomaly_scores[key_name] = pd.concat([drug_df, control_df], axis=0)
                    result_lines.append(f"{time}h - {control_name} vs {key_name}: 平均异常分数 = {mean_score:.4f}")

        self.result = "\n".join(result_lines)
        self.plot = self.draw_feature_histograms_with_pdf(feature_name)
        return self.anomaly_scores, self.result, self.plot

    def compute(self, control: pd.Series, drug: pd.Series) -> Tuple[float, pd.Series, pd.Series]:
        """
        计算异常分数：
        - 计算控制组ECDF
        - 对drug和control中每个点计算异常分数
        异常分数 = 1 - 2 * min(F(x), 1 - F(x))

        返回：
            - 平均异常分数（drug组）
            - drug组每个样本异常分数序列
            - control组每个样本异常分数序列
        """
        # 预处理，剔除NaN
        control = control.dropna()
        drug = drug.dropna()

        # 计算control的经验分布函数（ECDF）
        sorted_control = np.sort(control.values)

        def ecdf(x):
            # 返回x在control中的累积概率
            return np.searchsorted(sorted_control, x, side='right') / len(sorted_control)

        # 计算异常分数的向量化函数
        def anomaly_score(x):
            Fx = ecdf(x)
            return 1 - 2 * np.minimum(Fx, 1 - Fx)

        drug_scores = drug.apply(anomaly_score)
        control_scores = control.apply(anomaly_score)

        mean_score = float(drug_scores.mean())
        return mean_score, drug_scores, control_scores


    def draw_feature_histograms_with_pdf(self, feature_name):
        """
        统计self.data_df中不同时间点、处理组、浓度的feature_name分布，
        画直方图和核密度估计曲线，key格式为"{treatment}_{time}h_{conc}um"。

        返回：
            base64编码的PNG图片字符串
        """
        records = []  # 存储所有组的feature值及标签，用于合并绘图和统计

        # 提取独特的treatment, time, conc
        treatments = pd.unique(self.data['Metadata_treatment'])
        times = pd.unique(self.data['Metadata_hour'])
        concentrations = pd.unique(self.data['Metadata_concentration'])  # 假设这一列名是这个，你实际替换

        # 用于统一x轴范围
        all_feature_vals = self.data[feature_name][self.data[feature_name] > 0]
        if len(all_feature_vals) == 0:
            raise ValueError("feature_name中无有效数据")
        x_min, x_max = all_feature_vals.min(), all_feature_vals.max()
        x_range = np.linspace(x_min, x_max, 1000)

        plt.figure(figsize=(16, 10))
        colors = sns.color_palette("tab20", n_colors=len(treatments) * len(concentrations))

        color_idx = 0
        legend_labels = []

        for treatment in treatments:
            for time in times:
                for conc in concentrations:
                    # 筛选对应子集
                    subset = self.data[
                        (self.data['Metadata_treatment'] == treatment) &
                        (self.data['Metadata_hour'] == time) &
                        (self.data['Metadata_concentration'] == conc)
                        ]

                    vals = subset[feature_name].values[subset[feature_name] > 0]
                    if len(vals) < 2:
                        # 数据不足无法绘制核密度或直方图
                        continue

                    keyname = f"{treatment}_{time}h_{conc}um"
                    legend_labels.append(keyname)

                    # 画直方图（概率密度）
                    plt.hist(vals, bins=30, density=True, alpha=0.3,
                             color=colors[color_idx], label=None)

                    # 计算核密度估计
                    kde = gaussian_kde(vals)
                    pdf_vals = kde(x_range)

                    plt.plot(x_range, pdf_vals, color=colors[color_idx], lw=2, label=keyname)
                    color_idx += 1

        plt.xlabel(feature_name)
        plt.ylabel('概率密度')
        plt.title(f'{feature_name} 各时间点、处理组和浓度的分布直方图及核密度估计')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f'data:image/png;base64,{image_base64}'


if __name__ == '__main__':
    data_df = pd.read_csv(r"C:/Code/python/csv_data/gl/20250412/BCLXL-BAK/FRET替换513对照组数据.csv")
    sd_model = AnomalyScore(data_df)
    # Cell_Ed_region_top_50_value
    values, result_str, image = sd_model.start(feature_name='Cell_Ed_region_top_50_value')
    save_base64_with_prefix(image, r"C:\Users\pengs\Downloads\test.png")
