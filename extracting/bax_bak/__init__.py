import os
import numpy as np
import pandas as pd
from tifffile import tifffile

from extracting.bax_bak.colocalization import count_single_cell_localization
from extracting.bax_bak.rc import count_single_cell_rc
from extracting.bax_bak.ed import count_single_cell_Ed


def start(fret):
    """
    fret 具有如下参数可以调用
        self.current_sub_path = ''                  # 当前处理的子文件夹
        self.fret_target_name = fret_target_name    # FRET 靶点名称
        self.rc_min = rc_min
        self.rc_max = rc_max
        self.ed_min = ed_min
        self.ed_max = ed_max
        self.image_AA = None
        self.image_DD = None
        self.image_DA = None
        self.mask = None
        self.image_Ed = None
        self.image_Rc = None
        self.fret_mask = None
        self.nuclei_mask = None
        # 统计三通道的背景阈值
        self.background_noise_values = {}

    提取特征：
    1. 效率特征
    2. 浓度特征
    3. 共定位特征

    需要判断是否存在细胞核图像，决定是否掩码细胞核提取特征
    """
    cell_ed_df = pd.DataFrame()
    cell_rc_df = pd.DataFrame()
    rc_ed_df = None
    cell_localization_df = pd.DataFrame()
    if fret.need_Ed:
        cell_ed_df, regions_mask = count_single_cell_Ed(image_ed=fret.image_Ed.numpy(),
                                          image_rc=fret.image_Rc.numpy(),
                                          image_dd=fret.image_DD.numpy(),
                                          image_aa=fret.image_AA.numpy(),
                                          image_da=fret.image_DA.numpy(),
                                          background_noise_values = fret.background_noise_values,
                                          mask=fret.fret_mask.numpy(),
                                          rc_max=fret.rc_max,
                                          rc_min=fret.rc_min,
                                          ed_min=fret.ed_min,
                                          ed_max=fret.ed_max)
        # 保存区域掩码结果
        tifffile.imwrite(os.path.join(fret.current_sub_path, 'regions_mask.tif'), regions_mask * 255)
        cell_ed_df['ObjectNumber'] = cell_ed_df.index
        columns = ['ObjectNumber'] + [col for col in cell_ed_df.columns if col != 'ObjectNumber']
        # 按新顺序重新排列列
        cell_ed_df = cell_ed_df.reindex(columns=columns)
        print("效率特征")
    if fret.need_Rc:
        cell_rc_df, rc_ed_df = count_single_cell_rc(cell_mask=fret.fret_mask.numpy(),
                                                    regions_mask=regions_mask,
                                                    image_rc=fret.image_Rc.numpy(),
                                                    image_ed=fret.image_Ed.numpy())
        # 保存rc-ed的结果值
        rc_ed_df.to_csv(os.path.join(fret.current_sub_path, 'rc-ed.csv'))
        print("浓度特征")

    if fret.need_Fp:
        # 提取共定位信息
        cell_localization_df = count_single_cell_localization(image_dd=fret.image_DD.numpy(),
                                       image_aa=fret.image_AA.numpy(),
                                       image_da=fret.image_DA.numpy(),
                                       mask=fret.fret_mask.numpy(),
                                       regions_mask=regions_mask)
        # 提取溶度比信息，获取浓度比对应的rc-ed图像
        print("共定位特征")

    # 直接按列合并
    merged_df = pd.concat([cell_ed_df, cell_localization_df, cell_rc_df], axis=1)

    return merged_df, rc_ed_df


def process_masks(mit_mask, nuclei_mask):
    """
    处理两张掩码图像，删除 nmask 中mit_mask不存在的区域，并重新编码 nmask ，
    同时生成 nmask 掩码后的mit图像。

    :param mit_mask: 从1开始编码细胞区域的掩码图像
    :param nuclei_mask: 从1开始编码细胞区域的掩码图像
    :return: 处理后的 nmask 和 nmask 掩码后的mit图像
    """
    # 找出mit_mask中存在的区域编号
    mit_regions = np.unique(mit_mask)
    mit_regions = mit_regions[mit_regions > 0]  # 排除背景（编号为0）

    # 创建一个映射字典，将nmask中的区域编号映射到mit_mask中的编号
    region_mapping = {}
    new_nuclei_mask = np.zeros_like(nuclei_mask)
    new_index = 1

    for region in mit_regions:
        # 找出mit_mask中当前区域的位置
        mit_region_mask = (mit_mask == region)

        # 找出nmask中与当前mit区域重叠的区域
        overlapping_regions = np.unique(nuclei_mask[mit_region_mask])
        overlapping_regions = overlapping_regions[overlapping_regions > 0]  # 排除背景

        for overlap in overlapping_regions:
            if overlap not in region_mapping:
                region_mapping[overlap] = new_index
                new_nuclei_mask[nuclei_mask == overlap] = new_index
                new_index += 1

    # 生成nmask掩码后的mit图像
    mit_masked_by_nuclei_mask = np.where(new_nuclei_mask == 0, mit_mask, 0)

    return new_nuclei_mask, mit_masked_by_nuclei_mask