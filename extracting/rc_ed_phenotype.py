import os

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union


class CSVFeatureMerger:
    """用于将FRET特征文件中的特定列匹配并合并到多个表型特征文件中的工具"""

    def __init__(
            self,
            phenotype_dir: str,
            fret_file: str,
            output_dir: str,
            match_columns=None,
            fret_columns=None,
            file_pattern: str = "*.csv",
            verbose: bool = True
    ):
        """
        初始化CSV特征合并器

        参数:
            phenotype_dir: 表型特征文件所在目录
            fret_file: FRET特征文件路径
            output_dir: 输出目录
            match_columns: 用于匹配的列名列表
            fret_columns: 需要从FRET文件中提取的列名列表
            file_pattern: 表型特征文件的匹配模式
            verbose: 是否显示详细日志
        """
        if fret_columns is None:
            fret_columns = [
                "Ed_region_mean", "Rc_region_mean", 'Fp_region_PCC', "FRET_Judge", "Near_Ed"
            ]
        if match_columns is None:
            # TODO 还可以添加 Metadata_dish
            match_columns = [
                "Metadata_hour", "Metadata_treatment", "Metadata_site",
                "Metadata_concentration", "ObjectNumber"
            ]
        self.phenotype_dir = phenotype_dir
        self.fret_file = fret_file
        self.output_dir = output_dir
        self.match_columns = match_columns
        self.fret_columns = fret_columns
        self.file_pattern = file_pattern
        self.verbose = verbose

        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 存储FRET数据的DataFrame
        self.fret_data = None

    def load_fret_data(self) -> None:
        """加载FRET特征文件数据"""
        try:
            self.fret_data = pd.read_csv(self.fret_file)
            if self.verbose:
                print(f"已加载FRET特征文件: {self.fret_file}")
                print(f"FRET数据形状: {self.fret_data.shape}")
        except Exception as e:
            print(f"加载FRET特征文件时出错: {e}")
            self.fret_data = None

    def process_phenotype_file(self, phenotype_file: str) -> Optional[pd.DataFrame]:
        """
        处理单个表型特征文件

        参数:
            phenotype_file: 表型特征文件路径

        返回:
            合并后的DataFrame，如果出错则返回None
        """
        if self.fret_data is None:
            print("FRET数据未加载，请先调用load_fret_data()方法")
            return None

        try:
            # 读取表型特征文件
            phenotype_data = pd.read_csv(phenotype_file)
            if self.verbose:
                print(f"\n处理表型特征文件: {os.path.basename(phenotype_file)}")
                print(f"表型数据形状: {phenotype_data.shape}")

            # 验证匹配列是否存在
            missing_match_cols = [col for col in self.match_columns
                                  if col not in phenotype_data.columns]
            if missing_match_cols:
                print(f"错误: 表型特征文件缺少匹配列: {', '.join(missing_match_cols)}")
                return None

            # 验证FRET列是否存在
            missing_fret_cols = [col for col in self.fret_columns
                                 if col not in self.fret_data.columns]
            if missing_fret_cols:
                print(f"错误: FRET特征文件缺少列: {', '.join(missing_fret_cols)}")
                return None

            # 执行匹配合并
            merged_data = pd.merge(
                phenotype_data,
                self.fret_data[self.match_columns + self.fret_columns],
                on=self.match_columns,
                how='left'
            )

            if self.verbose:
                print(f"合并后数据形状: {merged_data.shape}")
                # 统计匹配成功的行数
                matched_rows = merged_data[self.fret_columns[0]].notna().sum()
                print(f"成功匹配: {matched_rows}/{len(merged_data)} 行")

            return merged_data

        except Exception as e:
            print(f"处理表型特征文件时出错: {e}")
            return None

    def save_merged_data(self, merged_data: pd.DataFrame, output_file: str) -> None:
        """
        保存合并后的数据到CSV文件

        参数:
            merged_data: 合并后的DataFrame
            output_file: 输出文件路径
        """
        ##############################################
        # 单细胞 FRET表征值计算
        ##############################################
        # 在这里计算对应的 Ed 效率表征值核 Fp 共定位表征值
        control_df = merged_data[merged_data['Metadata_treatment'] == 'control']
        control_mean = control_df['Ed_region_mean'].mean()
        control_std = control_df['Ed_region_mean'].std()
        z_score = (merged_data['Ed_region_mean'] - control_mean) / control_std
        # 线性映射方法
        z_abs = np.maximum(np.abs(z_score) - 0.5, 0)  # 减去阈值后再归一化
        merged_data['E_Ed'] = np.clip(z_abs / 2.5, 0, 1)  # 映射剩余部分
        # 限定最大差异为 z=3，超过的当成完全不相关
        merged_data['E_Fp'] = 1 - abs(merged_data['Fp_region_PCC'])

        # 计算最终的FRET表征值
        merged_data['E'] = 0.5 * merged_data['E_Ed'] + 0.5 * merged_data['E_Fp']
        try:
            merged_data.to_csv(output_file, index=False)
            if self.verbose:
                print(f"已保存合并后的数据到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def process_all_files(self) -> Dict[str, str]:
        """
        处理所有表型特征文件

        返回:
            包含处理结果的字典，键为输入文件名，值为输出文件名
        """
        if self.fret_data is None:
            self.load_fret_data()
            if self.fret_data is None:
                return {}

        results = {}

        # 获取所有表型特征文件
        phenotype_files = list(Path(self.phenotype_dir).glob(self.file_pattern))

        if not phenotype_files:
            print(f"未找到匹配的表型特征文件: {self.phenotype_dir}/{self.file_pattern}")
            return results

        if self.verbose:
            print(f"\n找到 {len(phenotype_files)} 个表型特征文件")

        # 处理每个表型特征文件
        for file_path in phenotype_files:
            file_name = os.path.basename(file_path)
            output_file = os.path.join(self.output_dir, file_name)

            # 处理文件
            merged_data = self.process_phenotype_file(str(file_path))

            # 保存结果
            if merged_data is not None:
                self.save_merged_data(merged_data, output_file)
                results[str(file_path)] = output_file

        return results


def merge_feature_files(
        phenotype_dir: str,
        fret_file: str,
        output_dir: str,
        match_columns: Optional[List[str]] = None,
        fret_columns: Optional[List[str]] = None,
        file_pattern: str = "*.csv",
        verbose: bool = True
) -> Dict[str, str]:
    """
    合并特征文件的主函数

    参数:
        phenotype_dir: 表型特征文件所在目录
        fret_file: FRET特征文件路径
        output_dir: 输出目录
        match_columns: 用于匹配的列名列表
        fret_columns: 需要从FRET文件中提取的列名列表
        file_pattern: 表型特征文件的匹配模式
        verbose: 是否显示详细日志

    返回:
        包含处理结果的字典，键为输入文件名，值为输出文件名
    """
    # 设置默认匹配列
    if match_columns is None:
        match_columns = [
            "Metadata_hour", "Metadata_treatment", "Metadata_site",
            "Metadata_concentration", "ObjectNumber"
        ]

    # 设置默认FRET列
    if fret_columns is None:
        fret_columns = [
            "Ed_region_mean", "Rc_region_mean", 'Fp_region_PCC', 'FRET_Judge', "Near_Ed"
        ]

    # 创建并运行合并器
    merger = CSVFeatureMerger(
        phenotype_dir=phenotype_dir,
        fret_file=fret_file,
        output_dir=output_dir,
        match_columns=match_columns,
        fret_columns=fret_columns,
        file_pattern=file_pattern,
        verbose=verbose
    )

    return merger.process_all_files()


if __name__ == "__main__":
    folder_path = r'C:\Code\python\csv_data\gl\BCLXL-BAX实验数据\20250513\BCLXL-BAK'
    # 使用示例
    PHENOTYPE_DIR = f"{folder_path}\表型表征值"  # 表型特征文件目录
    FRET_FILE = f"{folder_path}\Rc-Ed_FRET_analyzed.csv"  # FRET特征文件路径
    OUTPUT_DIR = r"C:\Users\pengs\Downloads"  # 输出目录

    # 运行合并操作
    results = merge_feature_files(
        phenotype_dir=PHENOTYPE_DIR,
        fret_file=FRET_FILE,
        output_dir=OUTPUT_DIR
    )

    # 打印结果摘要
    if results:
        print(f"\n成功处理 {len(results)} 个文件")
    else:
        print("\n未处理任何文件或处理过程中出错")