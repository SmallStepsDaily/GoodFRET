import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
"""
调用保存的模型计算对应的表型变化值
"""

class LDAInference:
    def __init__(self, csv_file, model_dir, ptype):
        df = pd.read_csv(csv_file)
        # 移除列名中的后缀（如 .1, .2 等）
        df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
        # 此时列名已重复，保留第一列并删除后续重复列名
        self.df = df.loc[:, ~df.columns.duplicated(keep='first')]
        self.model_dir = model_dir
        self.ptype = ptype
        self.result_dfs = {}
        self.result_str = ''
        self.feature_cols = [col for col in self.df.columns if
                        not col.startswith(
                            'Metadata_') and col != 'ObjectNumber' and col != 'Label' and col != 'ImageNumber']
        self.meta_cols = [col for col in self.df.columns if col.startswith('Metadata_') or col == 'ObjectNumber']
        print(self.meta_cols)

    def load_model(self, cell, treatment):
        model_name = f"{self.ptype}-{cell}-{treatment}.pkl"
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(model_path):
            return joblib.load(model_path)  # 返回 (model, scaler)
        else:
            print(f"模型不存在: {model_path}")
            return None, None

    def compute_phenotypic_value(self, df, drug_name):
        control_mean = df.loc[df['Metadata_treatment'] == 'control', 'Predicted_Probability'].mean()
        drug_probs = df.loc[(df['Metadata_treatment'] == drug_name), 'Predicted_Probability']
        drug_val = (drug_probs - control_mean).mean()
        df['S'] = df['Predicted_Probability'] - control_mean
        return df, drug_val

    def statistic_treatment_time_concentration(self, df):
        sns.set_theme(style="whitegrid", font_scale=1.2)
        df['Treatment_Concentration'] = df['Metadata_treatment'] + '_' + df['Metadata_concentration'].astype(str) + 'μM'

        tc_order = ['control_' + str(c) + 'μM' for c in
                    sorted(df[df['Metadata_treatment'] == 'control']['Metadata_concentration'].unique())]
        other_tc = [f'{t}_{c}μM' for t in sorted(df[df['Metadata_treatment'] != 'control']['Metadata_treatment'].unique())
                    for c in sorted(df[df['Metadata_treatment'] == t]['Metadata_concentration'].unique())]
        tc_order.extend(other_tc)
        df['Treatment_Concentration'] = pd.Categorical(df['Treatment_Concentration'], categories=tc_order, ordered=True)

        hour_order = sorted(df['Metadata_hour'].unique())
        df['Metadata_hour'] = pd.Categorical(df['Metadata_hour'], categories=hour_order, ordered=True)

        fig, ax = plt.subplots(figsize=(18, 10))
        sns.boxplot(
            x='Metadata_hour',
            y='S',
            hue='Treatment_Concentration',
            data=df,
            palette='tab20',
            ax=ax,
            width=0.8,
            showfliers=False
        )
        sns.stripplot(
            x='Metadata_hour',
            y='S',
            hue='Treatment_Concentration',
            data=df,
            palette='tab20',
            ax=ax,
            jitter=True,
            dodge=True,
            size=3,
            alpha=0.4
        )

        ax.set_title(f'{self.ptype} S Value Box Plot by Time, Treatment and Concentration', fontsize=16)
        ax.set_xlabel('Time (h)', fontsize=14)
        ax.set_ylabel('S Value', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(dict.fromkeys(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        plt.legend(unique_handles, unique_labels, title='Treatment & Concentration',
                   loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=True)

        plt.tight_layout()
        plt.subplots_adjust(right=0.8)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()
        return image

    def run(self, save_dir):
        unique_hours = self.df['Metadata_hour'].unique()
        unique_treatments = [t for t in self.df['Metadata_treatment'].unique() if t != 'control']

        cell = self.df['Metadata_cell'].iloc[0]
        all_results = []

        for treatment in unique_treatments:
            model, scaler = self.load_model(cell, treatment)
            if model is None:
                continue
            for hour in unique_hours:
                subset = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == treatment)]
                if subset.empty:
                    continue

                control_subset = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == 'control')]
                if control_subset.empty:
                    control_subset = self.df[self.df['Metadata_treatment'] == 'control']

                concentrations = subset['Metadata_concentration'].unique()

                for conc in concentrations:
                    conc_subset = subset[subset['Metadata_concentration'] == conc]
                    if conc_subset.empty:
                        continue

                    combined = pd.concat([conc_subset, control_subset])

                    X = combined[self.feature_cols]

                    X_scaled = scaler.transform(X)
                    probs = model.predict_proba(X_scaled)

                    classes = model.classes_ if hasattr(model, 'classes_') else None
                    if classes is None:
                        classes = model.base_estimator.classes_ if hasattr(model, 'base_estimator') else None
                    if classes is None:
                        print("无法获取模型类别")
                        continue
                    drug_idx = list(classes).index(treatment)

                    keep_df = combined[self.meta_cols]

                    keep_df = keep_df.reset_index(drop=True)
                    keep_df['Predicted_Probability'] = probs[:, drug_idx]

                    keep_df, drug_val = self.compute_phenotypic_value(keep_df, treatment)
                    all_results.append(keep_df)

                    key_name = f"{treatment}_{hour}h_{conc}um"
                    self.result_dfs[key_name] = combined

                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{self.ptype}-{key_name}.csv")
                    keep_df.to_csv(save_path, index=False)
                    print(f"保存文件: {save_path}")
                    self.result_str += f'在时间 {hour}h 中 {treatment} 加药组，浓度为 {conc}μm 的 {self.ptype} 药效表征值为 {drug_val}\n'


        # 合并所有结果用于绘图
        all_df = pd.concat(all_results, ignore_index=True)
        image = self.statistic_treatment_time_concentration(all_df)
        # 保存箱型图
        plot_path = os.path.join(save_dir, f"{self.ptype}-S_value_boxplot.png")
        image.save(plot_path)
        print(f"保存箱型图: {plot_path}")

        # 保存表征值结果
        with open(os.path.join(save_dir, f"{self.ptype}-表型表征值.txt"), 'w') as file:
            file.write(self.result_str)
            print("成功保存结果:", os.path.join(save_dir, "表型表征值.txt"))

        return self.result_dfs, image


if __name__ == '__main__':
    csv_path = r"C:\Code\python\csv_data\gl\EGFR-GRB2实验数据\H1975\20250618\FB_BF.csv"
    model_dir = 'model'
    ptype = 'BF'
    save_dir = r"C:\Users\pengs\Downloads/表型表征值"

    lda_infer = LDAInference(csv_path, model_dir, ptype)
    lda_infer.run(save_dir)
