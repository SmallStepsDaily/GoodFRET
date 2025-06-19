import os
import re
import threading
import pandas as pd
from batch.file import list_immediate_subdirectories, list_numeric_subdirectories
from extracting.compute import FRETComputer


def parse_batch_dir_string(input_string):
    """
    从给定的字符串中解析出 cell, treatment 和 hour 的值。
    支持小时部分为整数或浮点数。

    :param input_string: 输入字符串，格式为 'Cell-Treatment-Hour-d(Dish)-c(Concentration)ml'
    :return: 包含 cell, treatment 和 hour 、dish、concentration键值对的字典
    """
    # 定义正则表达式模式，支持浮点数
    pattern = r'^(?P<cell>[A-Za-z0-9]+)-(?P<treatment>[A-Za-z0-9]+)-(?P<hour>\d+(\.\d+)?)h-d(?P<dish>\d{1,2})-c(?P<concentration>\d+(\.\d+)?)μm$'

    match = re.match(pattern, input_string)

    if match:
        result = match.groupdict()
        # 添加带有 Metadata_ 前缀的新键，并转换数据类型
        metadata = {
            'Metadata_cell': result['cell'],
            'Metadata_treatment': result['treatment'],
            'Metadata_hour': float(result['hour']) if result['hour'] else None,
            'Metadata_dish': int(result['dish']) if result['dish'] else None,
            'Metadata_concentration': float(result['concentration']) if result['concentration'] else None
        }
        return metadata
    else:
        raise ValueError(f"输入字符串 '{input_string}' 不符合预期格式。")


class BatchProcessing:
    """
    批处理流程
    """

    def __init__(self, root, stop_event=threading.Event(), csv_name='FRET.csv'):
        # 单个批次文件路径
        self.root = root
        self.batch_dir_list = []
        self.current_Metadata_cell = ''
        self.current_Metadata_site = ''
        self.current_Metadata_hour = 0
        self.current_Metadata_treatment = ''
        self.current_Metadata_dish = 0
        self.current_Metadata_concentration = 0
        self.current_image_set_path = ''
        # 最后的批文件
        self.current_batch_data_df = pd.DataFrame()
        self.current_batch_rc_ed_df = pd.DataFrame()
        # 添加 stop_event 属性
        self.stop_event = stop_event
        # 保存的文件名称
        self.csv_name = csv_name

    def start(self, process_function, *args, **kwargs):
        """
        开始函数
        """
        self.batch_dir_list = list_immediate_subdirectories(self.root)
        # 遍历同个实验下不同批次文件
        for batch_dir in self.batch_dir_list:
            if self.stop_event.is_set():  # 检查是否需要停止
                break
            Metadata = parse_batch_dir_string(batch_dir)
            self.current_Metadata_cell = Metadata['Metadata_cell']
            self.current_Metadata_hour = Metadata['Metadata_hour']
            self.current_Metadata_treatment = Metadata['Metadata_treatment']
            self.current_Metadata_dish = Metadata['Metadata_dish']
            self.current_Metadata_concentration = Metadata['Metadata_concentration']
            # 获取文件夹下所有的视野文件夹列表
            batch_dir_path = str(os.path.join(self.root, batch_dir))
            batch_site_dir_list = list_numeric_subdirectories(batch_dir_path)

            # 遍历批次文件夹下的不同视野文件
            for batch_site_dir in batch_site_dir_list:
                # 检查是否需要停止
                if self.stop_event.is_set():
                    break
                site_dir_path = str(os.path.join(batch_dir_path, batch_site_dir))
                # 验证文件完整性
                self.current_Metadata_site = int(batch_site_dir)
                # 开始业务处理
                result = process_function(site_dir_path, *args, **kwargs)
                # 动态解包
                if isinstance(result, tuple):
                    # 处理多个返回值
                    result_df, rc_ed_df = result
                else:
                    # 处理单个返回值
                    result_df = result
                    rc_ed_df = None
                if result_df is not None:
                    # 保存基本信息到原有的文件上
                    result_df['Metadata_cell'] = self.current_Metadata_cell
                    result_df['Metadata_hour'] = self.current_Metadata_hour
                    result_df['Metadata_treatment'] = self.current_Metadata_treatment
                    result_df['Metadata_site'] = self.current_Metadata_site
                    result_df['Metadata_dish'] = self.current_Metadata_dish
                    result_df['Metadata_concentration'] = self.current_Metadata_concentration

                    # 调整列顺序，将 Metadata_ 开头的列放在最前面
                    metadata_cols = [col for col in result_df.columns if col.startswith('Metadata_')]
                    other_cols = [col for col in result_df.columns if not col.startswith('Metadata_')]
                    new_col_order = metadata_cols + other_cols
                    result_df = result_df[new_col_order]
                    # 将 Ed_df 拼接到当前批次的数据上
                    self.current_batch_data_df = pd.concat([self.current_batch_data_df, result_df], ignore_index=True)
                # 二级文件，如果返回None则不执行保存报错
                if rc_ed_df is not None:
                    # 保存基本信息到原有的文件上
                    rc_ed_df['Metadata_cell'] = self.current_Metadata_cell
                    rc_ed_df['Metadata_hour'] = self.current_Metadata_hour
                    rc_ed_df['Metadata_treatment'] = self.current_Metadata_treatment
                    rc_ed_df['Metadata_site'] = self.current_Metadata_site
                    rc_ed_df['Metadata_dish'] = self.current_Metadata_dish
                    rc_ed_df['Metadata_concentration'] = self.current_Metadata_concentration

                    # 调整列顺序，将 Metadata_ 开头的列放在最前面
                    metadata_cols = [col for col in rc_ed_df.columns if col.startswith('Metadata_') or col == 'ObjectNumber']
                    other_cols = [col for col in rc_ed_df.columns if not col.startswith('Metadata_')]
                    new_col_order = metadata_cols + other_cols
                    rc_ed_df = rc_ed_df[new_col_order]
                    # 将 Ed_df 拼接到当前批次的数据上
                    self.current_batch_rc_ed_df = pd.concat([self.current_batch_rc_ed_df, rc_ed_df], ignore_index=True)

        # 只有当 current_batch_data_df 不为空时才保存结果
        self.save_result()
        print(f"完成批处理操作 ========================================================> {self.root}")

    def save_result(self):
        """
        保存结果
        """
        if not self.current_batch_data_df.empty:
            self.current_batch_data_df.to_csv(os.path.join(self.root, self.csv_name), index=False)
        if not self.current_batch_rc_ed_df.empty:
            self.current_batch_rc_ed_df.to_csv(os.path.join(self.root, 'rc_ed.csv'), index=False)


if __name__ == "__main__":
    def process(image_set_path, fret_model):
        #############################
        # EGFR-FRET分析流程
        #############################
        # 进行分割流程
        return fret_model.start(image_set_path)


    fret = FRETComputer('bax_bak')
    batch = BatchProcessing(r'D:\data\20250513\BCLXL-BAK')
    batch.start(process, fret)