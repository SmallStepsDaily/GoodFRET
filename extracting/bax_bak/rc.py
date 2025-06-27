"""
统计rc值，计算rc和ed之间的函数关系
"""
import numpy as np
import pandas as pd

def count_single_cell_rc(cell_mask, regions_mask, image_rc, image_ed, need_Rc_Ed=False):
    """
    统计单细胞区域内的RC和ED指标

    参数:
    - cell_mask: 单细胞掩码，不同细胞从1开始编号
    - regions_mask: 单细胞内的区域划分掩码，与cell_mask对应
    - image_rc: RC值图像
    - image_ed: ED值图像

    返回:
    - cell_rc_df: 单细胞RC指标统计结果
    - rc_ed_df: RC-ED关系统计结果
    """
    # 确保输入数据维度一致
    assert cell_mask.shape == regions_mask.shape == image_rc.shape == image_ed.shape, "输入数据维度不一致"

    # 获取所有细胞ID
    cell_ids = np.unique(cell_mask)
    cell_ids = cell_ids[cell_ids > 0]  # 排除背景(0)

    # 初始化结果列表
    cell_rc_data = []
    rc_ed_data = []
    # 1. 统计每个细胞的RC指标
    for cell_id in cell_ids:
        # 获取当前细胞的掩码
        cell = cell_mask == cell_id

        # 获取细胞内的特定区域
        region = np.logical_and(cell, regions_mask > 0)
        non_region = np.logical_and(cell, regions_mask == 0)

        # 计算RC指标
        cell_rc_values = image_rc[cell]
        region_rc_values = image_rc[region]
        non_region_rc_values = image_rc[non_region]

        # 确保区域不为空
        if np.sum(region) == 0:
            region_mean = np.nan
            region_max = np.nan
            region_min = np.nan
            region_var = np.nan
        else:
            region_mean = np.mean(region_rc_values)
            region_max = np.max(region_rc_values)
            region_min = np.min(region_rc_values)
            region_var = np.var(region_rc_values)

        if np.sum(non_region) == 0:
            non_region_mean = np.nan
            non_region_max = np.nan
            non_region_min = np.nan
            non_region_var = np.nan
        else:
            non_region_mean = np.mean(non_region_rc_values)
            non_region_max = np.max(non_region_rc_values)
            non_region_min = np.min(non_region_rc_values)
            non_region_var = np.var(non_region_rc_values)

        # 保存结果
        cell_rc_data.append({
            'Rc_cell_mean': np.mean(cell_rc_values),
            'Rc_cell_max': np.max(cell_rc_values),
            'Rc_cell_min': np.min(cell_rc_values),
            'Rc_cell_var': np.var(cell_rc_values),
            'Rc_region_mean': region_mean,
            'Rc_region_max': region_max,
            'Rc_region_min': region_min,
            'Rc_region_var': region_var,
            'Rc_non_region_mean': non_region_mean,
            'Rc_non_region_max': non_region_max,
            'Rc_non_region_min': non_region_min,
            'Rc_non_region_var': non_region_var
        })

        '''
        开始计算Rc-Ed之间的关联
        '''
        if need_Rc_Ed:
            cell_ed_values = image_ed[cell]
            region_ed_values = image_ed[region]
            MAX_RC_VALUE = np.max(cell_rc_values)
            MIN_RC_VALUE = np.min(cell_rc_values)
            if MAX_RC_VALUE > 10:
                MAX_RC_VALUE = 10
            # 在0-MAX_RC_VALUE范围内以0.05为步长
            for rc_value in np.arange(MIN_RC_VALUE, MAX_RC_VALUE + 0.05, 0.05):
                rc_value = round(rc_value, 2)  # 避免浮点数精度问题

                # 找出RC值在当前区间的像素
                # 由于是连续值，使用范围±0.05
                cell_pixels = np.logical_and(cell_rc_values >= rc_value - 0.025, cell_rc_values < rc_value + 0.025)
                region_pixels = np.logical_and(region_rc_values >= rc_value - 0.025, region_rc_values < rc_value + 0.025)
                Cell_pixels_sum = np.sum(cell_pixels)
                Region_pixels_sum = np.sum(region_pixels)
                if np.sum(cell_pixels) > 5:
                    # 如果该像素数量小于30在这个区间，说明是异常值
                    # 计算区域内的ED均值
                    if Region_pixels_sum >= 5:
                        region_ed_mean = np.mean(region_ed_values[region_pixels])
                    else:
                        region_ed_mean = np.nan

                    # 计算细胞内的ED均值
                    if Cell_pixels_sum >= 10:
                        cell_ed_mean = np.mean(cell_ed_values[cell_pixels])
                    else:
                        cell_ed_mean = np.nan
                    # 保存结果
                    rc_ed_data.append({
                        'ObjectNumber': cell_id,
                        'Rc': rc_value,
                        'Region_Ed': region_ed_mean,
                        'Region_pixels_sum': Region_pixels_sum,
                        'Cell_Ed': cell_ed_mean,
                        'Cell_pixels_sum': Cell_pixels_sum,
                    })

    # 创建cell_rc_df
    cell_rc_df = pd.DataFrame(cell_rc_data, index=cell_ids)
    if need_Rc_Ed and len(rc_ed_data) > 0:
        # 创建rc_ed_df
        rc_ed_df = pd.DataFrame(rc_ed_data)
        # print(rc_ed_df)
        return cell_rc_df, rc_ed_df
    return cell_rc_df, None


