import os
import numpy as np
import pandas as pd
from tifffile import tifffile

from extracting.bax_bak import count_single_cell_rc, count_single_cell_localization
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
    image_ed = fret.image_Ed.numpy()
    image_rc = fret.image_Rc.numpy()
    image_dd = fret.image_DD.numpy()
    image_aa = fret.image_AA.numpy()
    image_da = fret.image_DA.numpy()
    mask = fret.fret_mask.numpy()
    cell_ed_df, seeds_mask = count_single_cell_Ed(image_ed=image_ed,
                                      image_rc=image_rc,
                                      image_dd=image_dd,
                                      image_aa=image_aa,
                                      image_da=image_da,
                                      background_noise_values = fret.background_noise_values,
                                      mask=mask,
                                      rc_max=fret.rc_max,
                                      rc_min=fret.rc_min,
                                      ed_min=fret.ed_min,
                                      ed_max=fret.ed_max
                                      )

    cell_ed_df = cell_ed_df.add_prefix('Ed_')
    print("效率特征")

    # 设置共定位特征文件和rc特征文件
    cell_localization_df = pd.DataFrame()
    cell_rc_df = pd. DataFrame()
    rc_ed_df = None

    # 该通过亚细胞器区域进行划分的操作存在争议，后续验证 TODO
    # nuclei_seeds_mask = None
    # mit_seeds_mask = None
    # if fret.extract_organelle and os.path.exists(os.path.join(fret.current_sub_path, 'nmask.tif')):
    #     nuclei_mask = load_image_to_numpy(os.path.join(fret.current_sub_path, 'nmask.tif'), dtype=np.uint8)
    #     nuclei_mask, mit_mask = process_masks(fret.fret_mask.numpy(), nuclei_mask)
    #     print(f"细胞核区域数量{nuclei_mask.max()} 线粒体区域数量{mit_mask.max()}")
    #     nuclei_ed_df, nuclei_seeds_mask = count_single_cell_Ed(image_ed=image_ed,
    #                                         image_rc=image_rc,
    #                                         image_dd=image_dd,
    #                                         image_aa=image_aa,
    #                                         image_da=image_da,
    #                                         background_noise_values=fret.background_noise_values,
    #                                         mask=nuclei_mask,
    #                                         rc_max=fret.rc_max,
    #                                         rc_min=fret.rc_min,
    #                                         ed_min=fret.ed_min,
    #                                         ed_max=fret.ed_max
    #                                         )
    #     nuclei_ed_df = nuclei_ed_df.add_prefix("Nuclei_")
    #     mit_ed_df, mit_seeds_mask = count_single_cell_Ed(image_ed=image_ed,
    #                                      image_rc=image_rc,
    #                                      image_dd=image_dd,
    #                                      image_aa=image_aa,
    #                                      image_da=image_da,
    #                                      background_noise_values=fret.background_noise_values,
    #                                      mask=mit_mask,
    #                                      rc_max=fret.rc_max,
    #                                      rc_min=fret.rc_min,
    #                                      ed_min=fret.ed_min,
    #                                      ed_max=fret.ed_max
    #                                      )
    #     mit_ed_df = mit_ed_df.add_prefix("Mit_")
    #     merged_df = pd.concat([cell_ed_df, nuclei_ed_df, mit_ed_df], axis=1)
    #     merged_df['ObjectNumber'] = merged_df.index
    # else:
    #     merged_df = cell_ed_df
    # if nuclei_seeds_mask is None and mit_seeds_mask is None:
    #     # 保存对应的聚点掩码图像
    #     save_seeds_mask = seeds_mask * 125 + np.where(mask > 0, 1, 0) * 125
    # else:
    #     save_seeds_mask = seeds_mask * 50 + nuclei_seeds_mask * 50 + mit_seeds_mask * 60 + np.where(mask > 0, 1, 0) * 50

    # 保存种子点图像
    save_seeds_mask = seeds_mask * 125 + np.where(mask > 0, 1, 0) * 125
    tifffile.imwrite(os.path.join(fret.current_sub_path, 'seeds_mask.tif'), save_seeds_mask.astype(np.uint8))

    if fret.need_Rc:
        cell_rc_df, rc_ed_df = count_single_cell_rc(cell_mask=mask,
                                                    regions_mask=seeds_mask,
                                                    image_rc=image_rc,
                                                    image_ed=image_ed,
                                                    need_Rc_Ed=fret.need_Rc_Ed)
        if fret.need_Rc_Ed and rc_ed_df is not None:
            # 保存rc-ed的结果值
            rc_ed_df.to_csv(os.path.join(fret.current_sub_path, 'rc-ed.csv'), index=False)
        print("浓度特征")

    if fret.need_Fp:
        # 提取共定位信息
        cell_localization_df = count_single_cell_localization(image_dd=image_dd,
                                       image_aa=image_aa,
                                       image_da=image_da,
                                       mask=mask,
                                       regions_mask=seeds_mask)
        # 提取溶度比信息，获取浓度比对应的rc-ed图像
        print("共定位特征")


    # 直接按列合并
    merged_df = pd.concat([cell_ed_df, cell_localization_df, cell_rc_df], axis=1)
    # 设置保存文件
    merged_df['ObjectNumber'] = merged_df.index
    columns = ['ObjectNumber'] + [col for col in merged_df.columns if col != 'ObjectNumber']
    # 按新顺序重新排列列
    merged_df = merged_df.reindex(columns=columns)

    return merged_df, rc_ed_df


def process_masks(mit_mask, nuclei_mask):
    """
    处理两张掩码图像，确保每个线粒体区域对应一个细胞核区域，
    删除mit_mask中不存在对应细胞核的区域，并重新编码细胞核掩码，
    同时生成细胞核掩码后的线粒体图像。

    :param mit_mask: 从1开始编码线粒体区域的掩码图像
    :param nuclei_mask: 从1开始编码细胞核区域的掩码图像
    :return: 处理后的细胞核掩码和细胞核掩码后的线粒体图像
    """
    # 找出mit_mask中存在的区域编号
    mit_regions = np.unique(mit_mask)
    mit_regions = mit_regions[mit_regions > 0]  # 排除背景（编号为0）

    # 找出nuclei_mask中存在的区域编号
    nuclei_regions = np.unique(nuclei_mask)
    nuclei_regions = nuclei_regions[nuclei_regions > 0]  # 排除背景

    # 记录已经被分配的细胞核区域
    assigned_nuclei = set()

    # 创建一个映射字典，将nuclei_mask中的区域编号映射到新的连续编号
    region_mapping = {}
    new_nuclei_mask = np.zeros_like(nuclei_mask)
    new_index = 1

    # 为每个线粒体区域找到最佳匹配的细胞核区域
    for mit_region in mit_regions:
        # 获取当前线粒体区域的掩码
        mit_region_mask = (mit_mask == mit_region)

        # 计算该线粒体区域与每个细胞核区域的重叠程度
        best_overlap_ratio = 0
        best_nucleus = None

        for nucleus_region in nuclei_regions:
            # 跳过已经被分配的细胞核
            if nucleus_region in assigned_nuclei:
                continue

            # 获取当前细胞核区域的掩码
            nucleus_region_mask = (nuclei_mask == nucleus_region)

            # 计算重叠区域
            overlap = np.logical_and(mit_region_mask, nucleus_region_mask)

            # 计算重叠比例（相对于细胞核区域）
            overlap_ratio = np.sum(overlap) / np.sum(nucleus_region_mask) if np.sum(nucleus_region_mask) > 0 else 0

            # 如果这个细胞核区域与当前线粒体的重叠更好，则更新最佳匹配
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
                best_nucleus = nucleus_region

        # 如果找到了匹配的细胞核区域，并且重叠比例足够高，则建立映射
        if best_nucleus is not None and best_overlap_ratio > 0.1:  # 设置最小重叠阈值
            region_mapping[best_nucleus] = new_index
            new_nuclei_mask[nuclei_mask == best_nucleus] = new_index
            assigned_nuclei.add(best_nucleus)
            new_index += 1

    # 生成细胞核掩码后的线粒体图像
    # 只保留那些有线粒体-细胞核匹配的线粒体区域
    mit_masked_by_nuclei_mask = np.zeros_like(mit_mask)
    for old_nucleus, new_idx in region_mapping.items():
        nucleus_mask = (nuclei_mask == old_nucleus)
        # 找出与这个细胞核匹配的线粒体区域
        matched_mit_region = None
        for mit_region in mit_regions:
            mit_region_mask = (mit_mask == mit_region)
            overlap = np.logical_and(mit_region_mask, nucleus_mask)
            if np.sum(overlap) > 0:
                matched_mit_region = mit_region
                break

        if matched_mit_region is not None:
            # 只保留线粒体中与细胞核重叠的部分
            mit_masked_by_nuclei_mask[overlap] = matched_mit_region

    return new_nuclei_mask, mit_masked_by_nuclei_mask