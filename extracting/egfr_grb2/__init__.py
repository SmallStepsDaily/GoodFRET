import os
import numpy as np
import pandas as pd

from extracting.compute import load_image_to_numpy
from extracting.egfr_grb2.ed import count_single_cell_Ed


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
    cell_ed_df = count_single_cell_Ed(image_ed=fret.image_Ed.numpy(),
                                      image_rc=fret.image_Rc.numpy(),
                                      image_dd=fret.image_DD.numpy(),
                                      image_aa=fret.image_AA.numpy(),
                                      image_da=fret.image_DA.numpy(),
                                      background_noise_values = fret.background_noise_values,
                                      mask=fret.fret_mask.numpy(),
                                      rc_max=fret.rc_max,
                                      rc_min=fret.rc_min,
                                      ed_min=fret.ed_min,
                                      ed_max=fret.ed_max
                                      )
    cell_ed_df = cell_ed_df.add_prefix('Cell_')
    if fret.extract_organelle and os.path.exists(os.path.join(fret.current_sub_path, 'nmask.tif')):
        nuclei_mask = load_image_to_numpy(os.path.join(fret.current_sub_path, 'nmask.tif'), dtype=np.uint8)
        nuclei_mask, mit_mask = process_masks(fret.fret_mask.numpy(), nuclei_mask)
        nuclei_ed_df = count_single_cell_Ed(image_ed=fret.image_Ed.numpy(),
                                            image_rc=fret.image_Rc.numpy(),
                                            image_dd=fret.image_DD.numpy(),
                                            image_aa=fret.image_AA.numpy(),
                                            image_da=fret.image_DA.numpy(),
                                            background_noise_values=fret.background_noise_values,
                                            mask=nuclei_mask,
                                            rc_max=fret.rc_max,
                                            rc_min=fret.rc_min,
                                            ed_min=fret.ed_min,
                                            ed_max=fret.ed_max
                                            )
        nuclei_ed_df = nuclei_ed_df.add_prefix("Nuclei_")
        mit_ed_df = count_single_cell_Ed(image_ed=fret.image_Ed.numpy(),
                                         image_rc=fret.image_Rc.numpy(),
                                         image_dd=fret.image_DD.numpy(),
                                         image_aa=fret.image_AA.numpy(),
                                         image_da=fret.image_DA.numpy(),
                                         background_noise_values=fret.background_noise_values,
                                         mask=mit_mask,
                                         rc_max=fret.rc_max,
                                         rc_min=fret.rc_min,
                                         ed_min=fret.ed_min,
                                         ed_max=fret.ed_max
                                         )
        mit_ed_df = mit_ed_df.add_prefix("Mit_")
        merged_df = pd.concat([cell_ed_df, nuclei_ed_df, mit_ed_df], axis=1)
        merged_df['ObjectNumber'] = merged_df.index
    else:
        merged_df = cell_ed_df
    merged_df['ObjectNumber'] = merged_df.index
    return merged_df



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