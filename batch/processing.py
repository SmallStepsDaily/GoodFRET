import os
import re
import pandas as pd
from batch.file import list_immediate_subdirectories, list_numeric_subdirectories


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
    def __init__(self, root):
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

    def start(self, process_function, *args, **kwargs):
        """
        开始函数
        """
        self.batch_dir_list = list_immediate_subdirectories(self.root)
        # 遍历同个实验下不同批次文件
        for batch_dir in self.batch_dir_list:
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
                site_dir_path = str(os.path.join(batch_dir_path, batch_site_dir))
                # 验证文件完整性
                self.current_Metadata_site = int(batch_site_dir)
                # 开始业务处理
                process_function(site_dir_path, *args, **kwargs)
