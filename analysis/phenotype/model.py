import pandas as pd
import re

class Model:
    def __init__(self, df, ptype):
        # 删除完全由缺失值组成的列
        df = df.dropna(axis=1, how='all').copy()
        # 删除任何包含缺失值的行
        df_cleaned = df.dropna()
        # 特征合并操作
        df = self.merge_Texture_features(df_cleaned)

        # 定义df的特征类型
        self.ptype = ptype
        self.df = df
        # 去除 Metadata_ 开头和 ObjectNumber 的特征列表
        self.features_columns = [col for col in df.columns if
                                 not col.startswith('Metadata_') and col != 'ObjectNumber' and col != 'Label' and col != 'ImageNumber']

    @staticmethod
    def merge_Texture_features(data: pd.DataFrame):
        """
            合并 Texture 不同方向上的特征。

            参数:
            data (pd.DataFrame): 包含 Texture 特征的 DataFrame。

            返回:
            pd.DataFrame: 合并后的 DataFrame。
            """
        # 创建 DataFrame 的显式副本以避免修改原始数据
        data = data.copy()

        pattern = r'^(Texture_[A-Za-z0-9]+_[A-Za-z0-9]+_\d{1,2}_).+$'

        # 筛选符合条件的列名，并根据前缀分类
        merge_features = {}
        for col in data.columns:
            match = re.match(pattern, col)
            if match:
                prefix = match.group(1)  # 获取捕获组中的前缀
                if prefix not in merge_features:
                    merge_features[prefix] = []
                merge_features[prefix].append(col)

        # 准备要添加的新列和要删除的旧列
        new_columns = {}
        columns_to_drop = []

        # 特征均值计算
        for prefix, cols in merge_features.items():
            if cols:  # 确保列表不为空
                mean_col_name = f'{prefix}Mean'
                new_columns[mean_col_name] = data[cols].mean(axis=1)
                columns_to_drop.extend(cols)

        # 使用 pd.concat 一次性添加所有新列
        if new_columns:
            new_df = pd.DataFrame(new_columns)
            data = pd.concat([data, new_df], axis=1)

        # 删除所有旧列
        if columns_to_drop:
            data.drop(columns=columns_to_drop, inplace=True)

        return data