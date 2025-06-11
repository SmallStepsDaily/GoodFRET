import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from analysis.phenotype.loading import FileLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from analysis.phenotype.model import Model


class LDAClassifyModel(Model):
    def __init__(self, df, ptype):
        super().__init__(df, ptype)
        self.result_str = ""
        self.result_df = {}
        self.result_boxplot = None
        self.result_train_images = {}
        self._run()

    @staticmethod
    def compute(drug, control, features_columns, drug_name):
        """
        合并 drug 和 control 数据进行 LDA 分析
        输入1组加药数据和对照数据
        """
        combined_data = pd.concat([drug, control])
        X = combined_data[features_columns]
        y = combined_data['Metadata_treatment']

        # 保存所需的元数据列
        metadata_cols = ['ObjectNumber', 'Metadata_site', 'Metadata_concentration', 'Metadata_hour']
        metadata = combined_data[metadata_cols].copy()

        # 1. 划分数据集为训练集和验证集（例如 70% 训练，30% 测试）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        indices = np.argwhere(X_train == 'A1331852')
        print(indices)
        # 创建测试集标记
        is_test = pd.Series(False, index=X.index)
        is_test[X_test.index] = True

        # 2. 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. 创建并训练校准后的 LDA 模型
        lda = LinearDiscriminantAnalysis()
        calibrated_lda = CalibratedClassifierCV(lda, cv=5)
        calibrated_lda.fit(X_train_scaled, y_train)

        # 4. 在测试集上评估模型
        y_test_pred = calibrated_lda.predict(X_test_scaled)
        print(drug_name, "Validation Classification Report:")
        print(classification_report(y_test, y_test_pred))

        # 使用 LabelBinarizer 将多分类标签转换为二进制格式
        lb = LabelBinarizer()
        lb.fit(y_train)  # 确保使用训练集来拟合 LabelBinarizer

        # 获取正类的索引
        drug_label = list(lb.classes_).index(drug_name)

        # 标准化全部数据
        X_scaled = scaler.transform(X)

        # 预测全部数据的概率
        all_probabilities = calibrated_lda.predict_proba(X_scaled)[:, drug_label]

        # 创建包含所有数据的预测结果
        all_results = pd.DataFrame({
            'Metadata_treatment': y,
            'Predicted_Probability': all_probabilities,
            'Is_Test': is_test
        })

        # 合并保留的元数据列
        all_results = pd.concat([metadata, all_results], axis=1)

        # 计算训练集的 ROC 曲线
        y_train_binary = lb.transform(y_train)
        y_train_prob = calibrated_lda.predict_proba(X_train_scaled)[:, drug_label]
        fpr_train, tpr_train, _ = roc_curve(y_train_binary.ravel(), y_train_prob, pos_label=drug_label)
        roc_auc_train = auc(fpr_train, tpr_train)

        # 计算测试集的 ROC 曲线
        y_test_binary = lb.transform(y_test)
        y_test_prob = calibrated_lda.predict_proba(X_test_scaled)[:, drug_label]
        fpr_test, tpr_test, _ = roc_curve(y_test_binary.ravel(), y_test_prob, pos_label=drug_label)
        roc_auc_test = auc(fpr_test, tpr_test)

        # 创建包含三个子图的图表：ROC曲线、概率直方图、箱型图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # 子图 1: ROC 曲线
        ax1.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training ROC curve (AUC = {roc_auc_train:.2f})')
        ax1.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")

        # 子图 2: 概率值的统计信息（这里使用直方图）
        sns.histplot(data=all_results, x='Predicted_Probability', hue='Metadata_treatment', bins=20, kde=True, ax=ax2)
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Frequency')

        # 子图 3: 箱型图
        sns.boxplot(x='Metadata_treatment', y='Predicted_Probability', data=all_results, ax=ax3)
        ax3.set_title('Box Plot of Predicted Probabilities by True Label')
        ax3.set_xlabel('True Label')
        ax3.set_ylabel('Predicted Probability')

        plt.tight_layout()
        # 将图像保存到 BytesIO 对象
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        # 使用 PIL 打开图像
        image = Image.open(buf)
        # 关闭缓冲区和图形，以释放资源
        plt.close()

        return all_results, image

    @staticmethod
    def compute_phenotypic_value(df, drug_name):
        """
        计算表型表征值
        """
        control_predict_mean = df.loc[df['Metadata_treatment'] == 'control', 'Predicted_Probability'].mean()
        # 仅计算测试集的预测概率
        drug_predict_list = df.loc[(df['Metadata_treatment'] == drug_name) & (df['Is_Test'] == True), 'Predicted_Probability']
        drug_predict_value = (drug_predict_list - control_predict_mean).mean()
        # 计算表型表征值
        df['S'] = df['Predicted_Probability'] - control_predict_mean
        return df, drug_predict_value

    def save_dict_to_csv_files(self, save_path, ptype='BF'):
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
        for key, df in self.result_df.items():
            try:
                # 构建完整的文件路径
                file_path = os.path.join(save_path, f"{ptype}-{key}.csv")

                # 直接保存DataFrame
                df.to_csv(file_path, index=False)

                print(f"成功保存文件: {file_path}")

            except Exception as e:
                print(f"错误: 保存键 '{key}' 时出错 - {str(e)}")

    def save_dict_to_images(self, save_path, format='PNG', ptype='BF'):
        """
        将字典中的PIL Image对象保存为独立的图像文件

        参数:
        data_dict (dict): 键值对字典，键为文件名，值为PIL Image对象
        save_path (str): 保存图像文件的目标文件夹路径
        format (str): 图像格式，默认为PNG

        返回:
        None
        """
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 遍历字典中的每个项目
        for key, img in self.result_train_images.items():
            try:
                # 构建完整的文件路径（添加扩展名）
                file_path = os.path.join(save_path, f"{ptype}-{key}.{format.lower()}")

                # 保存图像
                img.save(file_path, format=format)

                print(f"成功保存图像: {file_path}")

            except Exception as e:
                print(f"错误: 保存键 '{key}' 时出错 - {str(e)}")

    def save_result_image(self, sava_path, save_name):
        file_path = os.path.join(sava_path, save_name)
        self.result_boxplot.save(file_path)
        print(f"成功保存图像: {file_path}")

    def _run(self):
        # 获取不同的 Metadata_hour 和 Metadata_treatment 组合
        unique_hours = self.df['Metadata_hour'].unique()
        # 去除 'control' 处理
        unique_treatments = [treatment for treatment in self.df['Metadata_treatment'].unique() if
                             treatment != 'control']
        result_str = ""
        result_df = {}
        image_df = pd.DataFrame()
        print(f"时间点数量: {unique_hours}")
        print(f"干扰组数量: {unique_treatments}")
        for hour in unique_hours:
            for treatment in unique_treatments:
                # 筛选出当前 Metadata_hour 和 Metadata_treatment 的数据
                subset_data = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == treatment)]

                # 提示当前时间点
                print(f"LDA计算时间为 {hour}h")

                # 如果该时间点下的加药组数据为空，则不执行后续操作
                if subset_data.empty:
                    continue

                # 筛选出 Metadata_hour 相同且 Metadata_treatment 为 control 的数据
                control_data = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == 'control')]

                # 假设当前时间点对照组为空，则选择任意时间节点的control数据
                if control_data.empty:
                    control_data = self.df[self.df['Metadata_treatment'] == 'control']
                # 遍历浓度进行计算
                concentrations = pd.unique(subset_data['Metadata_concentration'])
                for concentration in concentrations:
                    concentration_subset_data = subset_data[subset_data['Metadata_concentration'] == concentration]

                    if concentration_subset_data.empty:
                        continue
                    key_name = f'{treatment}_{hour}h_{concentration}um'
                    # 将两类数据输入到 computer 函数中
                    drug_predict_df, drug_predict_image = self.compute(concentration_subset_data, control_data, self.features_columns, treatment)
                    # 保存训练模型的loss图像
                    self.result_train_images[key_name] = drug_predict_image
                    # TODO 需要修改为每个加药情况单独一个统计
                    value_df, result_value = self.compute_phenotypic_value(drug_predict_df, treatment)
                    result_df[key_name] = value_df
                    result_str += f'在时间 {hour}h 中 {treatment} 加药组，浓度为 {concentration}μm 的 {self.ptype} 药效表征值为 {result_value}\n'
                    image_df = pd.concat([image_df, value_df.reset_index()], axis=0)
        boxplot_result = statistic_treatment_time_concentration(image_df, self.ptype)
        self.result_str = result_str
        self.result_df = result_df
        self.result_boxplot = boxplot_result


def statistic_treatment_time_concentration(df, feature_type):
    """
    统计不同时间、干扰和浓度的数据情况，并绘制箱型图。

    :param df: 包含 'Metadata_hour', 'S', 'Metadata_treatment' 和 'Metadata_concentration' 列的 DataFrame
    :param feature_type: 特征类型，可以是线粒体、细胞核或者明场图像
    :return: 返回绘制好的图像对象
    """
    # 设置绘图风格
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 创建treatment_concentration组合列
    df['Treatment_Concentration'] = df['Metadata_treatment'] + '_' + df['Metadata_concentration'].astype(str) + 'μM'

    # 确保control组排在前面
    tc_order = ['control_' + str(c) + 'μM' for c in
                sorted(df[df['Metadata_treatment'] == 'control']['Metadata_concentration'].unique())]
    other_tc = [f'{t}_{c}μM' for t in sorted(df[df['Metadata_treatment'] != 'control']['Metadata_treatment'].unique())
                for c in sorted(df[df['Metadata_treatment'] == t]['Metadata_concentration'].unique())]
    tc_order.extend(other_tc)

    df['Treatment_Concentration'] = pd.Categorical(df['Treatment_Concentration'], categories=tc_order, ordered=True)

    # 将时间列转换为有序分类变量
    hour_order = sorted(df['Metadata_hour'].unique())
    df['Metadata_hour'] = pd.Categorical(df['Metadata_hour'], categories=hour_order, ordered=True)

    # 创建一个更大的图形以适应复杂的可视化
    fig, ax = plt.subplots(figsize=(18, 10))

    # 使用hue参数表示treatment_concentration组合
    boxplot = sns.boxplot(
        x='Metadata_hour',
        y='S',
        hue='Treatment_Concentration',
        data=df,
        palette='tab20',  # 使用更丰富的调色板
        ax=ax,
        width=0.8,  # 调整箱型图宽度
        showfliers=False  # 不显示异常值，使图表更清晰
    )

    # 添加散点图展示原始数据分布
    strip = sns.stripplot(
        x='Metadata_hour',
        y='S',
        hue='Treatment_Concentration',
        data=df,
        palette='tab20',
        ax=ax,
        jitter=True,  # 添加抖动使点分布更清晰
        dodge=True,  # 使散点图按照hue分离
        size=3,  # 调整点的大小
        alpha=0.4  # 设置透明度
    )

    # 添加标题和标签
    ax.set_title(f'{feature_type} S Value Box Plot by Time, Treatment and Concentration', fontsize=16)
    ax.set_xlabel('Time (h)', fontsize=14)
    ax.set_ylabel('S Value', fontsize=14)

    # 美化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 创建自定义图例
    handles, labels = boxplot.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))  # 保持顺序去重
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # 调整图例位置
    plt.legend(unique_handles, unique_labels, title='Treatment & Concentration',
               loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, frameon=True)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # 为右侧的图例留出更多空间

    # 将图像保存到BytesIO对象
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    # 使用PIL打开图像
    image = Image.open(buf)

    # 关闭图形以释放资源
    plt.close()
    return image


def run_lda(file_paths, save_path):
    """
    file_paths 文件列表
    save_path 保存路径
    """
    files = FileLoader(file_paths)
    result_str = ""
    # 保存对应的结果
    if files.mit_df is not None:
        mit_result = LDAClassifyModel(files.mit_df, 'Mit')
        mit_result.save_dict_to_csv_files(save_path, ptype='Mit')
        mit_result.save_dict_to_images(save_path, ptype='Mit')
        mit_result.save_result_image(save_path, "线粒体荧光表征值箱型图.png")
        result_str += mit_result.result_str + '\n'
    if files.bf_df is not None:
        bf_result = LDAClassifyModel(files.bf_df, 'BF')
        bf_result.save_dict_to_csv_files(save_path, ptype='BF')
        bf_result.save_dict_to_images(save_path, ptype='BF')
        bf_result.save_result_image(save_path, "明场表征值箱型图.png")
        result_str += bf_result.result_str + '\n'
    if files.nuclei_df is not None:
        nuclei_result = LDAClassifyModel(files.nuclei_df, 'Nuclei')
        nuclei_result.save_dict_to_csv_files(save_path, ptype='Nuclei')
        nuclei_result.save_dict_to_images(save_path, ptype='Nuclei')
        nuclei_result.save_result_image(save_path, "细胞核荧光表征值箱型图.png")
        result_str += nuclei_result.result_str + '\n'

    # 保存表征值结果
    with open(os.path.join(save_path, "表型表征值.txt"), 'w') as file:
        file.write(result_str)
        print("成功保存结果:", os.path.join(save_path, "表型表征值.txt"))


if __name__ == '__main__':
    paths = [r"C:\Code\python\csv_data\gl\20250412\BCLXL-BAK\BCLXL-FB_BF四种实验数据.csv"]
    run_lda(paths, 'C:/Users/pengs/Downloads')