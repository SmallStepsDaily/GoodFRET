import os.path

import pandas as pd
import seaborn as sns
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from anaylsis.phenotype.loading import FileLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from anaylsis.phenotype.model import Model


class LDAClassifyModel(Model):
    def __init__(self, df, ptype):
        super().__init__(df, ptype)
        self.result_str = ""
        self.result_df = None
        self.result_boxplot = None
        self._run()

    @staticmethod
    def compute(drug, control, features_columns, drug_name):
        """
        合并 drug 和 control 数据进行 LDA 分析
        """
        combined_data = pd.concat([drug, control])
        X = combined_data[features_columns]
        y = combined_data['Metadata_treatment']

        # 1. 划分数据集为训练集、验证集和测试集（例如 60% 训练，40% 测试）
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

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

        # 保存预测概率到 CSV 文件
        test_results = pd.DataFrame({
            'Metadata_treatment': y_test,
            'Predicted_Probability': y_test_prob
        })

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
        sns.histplot(data=test_results, x='Predicted_Probability', hue='Metadata_treatment', bins=20, kde=True, ax=ax2)
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Frequency')

        # 子图 3: 箱型图
        # 准备数据：将预测的概率值按真实标签分组
        grouped_probabilities = test_results.groupby('Metadata_treatment')['Predicted_Probability'].apply(list).reset_index()
        grouped_probabilities.columns = ['Metadata_treatment', 'Probabilities']

        # 绘制箱型图
        sns.boxplot(x='Metadata_treatment', y='Predicted_Probability', data=test_results, ax=ax3)
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
        return test_results, image

    @staticmethod
    def compute_phenotypic_value(df, drug_name):
        """
        计算表型表征值
        """
        control_predict_mean = df.loc[df['Metadata_treatment'] == 'control', 'Predicted_Probability'].mean()
        drug_predict_list = df.loc[df['Metadata_treatment'] == drug_name, 'Predicted_Probability']
        drug_predict_value = (drug_predict_list - control_predict_mean).mean()
        df['Predicted_Probability'] = df['Predicted_Probability'] - control_predict_mean
        return df, drug_predict_value

    def _run(self):
        # 获取不同的 Metadata_hour 和 Metadata_treatment 组合
        unique_hours = self.df['Metadata_hour'].unique()
        # 去除 'control' 处理
        unique_treatments = [treatment for treatment in self.df['Metadata_treatment'].unique() if
                             treatment != 'control']
        result_str = ""
        result_df = pd.DataFrame()
        image_df = pd.DataFrame()
        for hour in unique_hours:
            for treatment in unique_treatments:
                # 筛选出当前 Metadata_hour 和 Metadata_treatment 的数据
                subset_data = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == treatment)]
                # 筛选出 Metadata_hour 相同且 Metadata_treatment 为 control 的数据
                control_data = self.df[(self.df['Metadata_hour'] == hour) & (self.df['Metadata_treatment'] == 'control')]
                if control_data.empty:
                    control_data = self.df[self.df['Metadata_treatment'] == 'control']
                if not subset_data.empty and not control_data.empty:
                    # 将两类数据输入到 computer 函数中
                    drug_predict_df,drug_predict_image = self.compute(subset_data, control_data, self.features_columns, treatment)

                    value_df, result_value = self.compute_phenotypic_value(drug_predict_df, treatment)
                    value_df['Metadata_hour'] = hour
                    # 横向拼接 DataFrame
                    if result_df.empty:
                        image_df = value_df
                        value_df = value_df.add_prefix(f'{treatment}_')
                        result_df = value_df.reset_index()
                    else:
                        image_df = pd.concat([image_df, value_df], axis=0)
                        # 为列名添加前缀
                        value_df = value_df.add_prefix(f'{treatment}_')
                        result_df = pd.concat([result_df, value_df.reset_index()], axis=1)

                    result_str += f'{treatment} 在时间 {hour}h 的 {self.ptype} 药效表征值为 {result_value}\n'
        # 进行图像的分析
        boxplot_result = statistic_treatment_and_time(image_df, self.ptype)
        self.result_str = result_str
        self.result_df = result_df
        self.result_boxplot = boxplot_result


def statistic_treatment_and_time(df, feature_type):
    """
    统计不同时间和干扰的数据情况，并绘制箱型图。

    :param df: 包含 'Metadata_hour', 'Predicted_Probability', 和 'Metadata_treatment' 列的 DataFrame
    :param feature_type: 特征类型，可以是线粒体、细胞核或者明场图像
    :return: 返回绘制好的图像对象
    """
    # 设置绘图风格
    sns.set(style="whitegrid")

    # 重新排序 Metadata_treatment 列，将 control 放在最前面
    treatment_order = ['control'] + [val for val in df['Metadata_treatment'].unique() if val != 'control']
    df['Metadata_treatment'] = pd.Categorical(df['Metadata_treatment'], categories=treatment_order, ordered=True)

    # 创建一个固定大小的图形和子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用 catplot 方法绘制箱型图，设置 kind 参数为 "box"
    g = sns.boxplot(
        x='Metadata_hour', y='Predicted_Probability', hue='Metadata_treatment',
        data=df,
        ax=ax
    )

    # 添加标题和标签
    ax.set_title(f'{feature_type} Phenotypic Value Box Plot')
    ax.set_xlabel('Time(h)')
    ax.set_ylabel('Phenotypic Value')

    # 规范图例标题
    plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # 调整布局以防止重叠
    plt.tight_layout()

    # 调整图例位置，确保其不在图表之外被裁剪
    plt.subplots_adjust(right=0.85)

    # 将图像保存到 BytesIO 对象
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)

    # 使用 PIL 打开图像
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
        mit_result.result_df.to_csv(os.path.join(save_path, "Mit_prediction.csv"))
        mit_result.result_boxplot.save(os.path.join(save_path, "线粒体荧光表征值箱型图.png"))
        result_str += mit_result.result_str + '\n'
    if files.bf_df is not None:
        bf_result = LDAClassifyModel(files.bf_df, 'BF')
        bf_result.result_df.to_csv(os.path.join(save_path, "BF_prediction.csv"))
        bf_result.result_boxplot.save(os.path.join(save_path, "明场表征值箱型图.png"))
        result_str += bf_result.result_str + '\n'
    if files.nuclei_df is not None:
        nuclei_result = LDAClassifyModel(files.nuclei_df, 'Nuclei')
        nuclei_result.result_df.to_csv(os.path.join(save_path, "Nuclei_prediction.csv"))
        nuclei_result.result_boxplot.save(os.path.join(save_path, "细胞核荧光表征值箱型图.png"))
        result_str += nuclei_result.result_str + '\n'

    # 保存表征值结果
    with open(os.path.join(save_path, "表型表征值.txt"), 'w') as file:
        file.write(result_str)


if __name__ == '__main__':
    paths = [r"C:\Code\python\csv_data\qrm\20250319_PC9_FOXO3A_4h\表型表征值\Foxo3a_BF.csv", r"C:\Code\python\csv_data\qrm\20250319_PC9_FOXO3A_4h\表型表征值\Foxo3a_Nuclei.csv"]
    run_lda(paths, 'C:/Users/pengs/Downloads')