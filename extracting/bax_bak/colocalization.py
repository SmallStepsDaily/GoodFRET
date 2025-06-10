"""
DD DA AA 通道共定位特征查看蛋白定位情况
"""
import numpy as np
import pandas as pd


def count_single_cell_localization(image_dd, image_da, image_aa, mask, regions_mask):
    """
    统计单细胞内不同区域的AA/DD强度比

    参数:
        image_dd: dd通道图像
        image_da: da通道图像
        image_aa: aa通道图像
        mask: 细胞掩码，背景为0，每个细胞区域为唯一整数
        regions_mask: 线粒体区域掩码，背景为0，线粒体区域为1

    返回:
        df: 统计结果DataFrame，包含各细胞的强度比信息
    """
    # 确保所有输入图像尺寸一致
    assert image_dd.shape == image_aa.shape == mask.shape == regions_mask.shape, "图像尺寸不一致"

    # 提取所有细胞的ID
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]  # 排除背景(0)

    # 初始化结果列表
    results = []

    # 遍历每个细胞
    for cell_id in cell_ids:
        # 提取当前细胞的区域
        cell_mask = (mask == cell_id).astype(np.uint8)

        # 计算当前细胞的各项统计数据
        stats = calculate_cell_stats(image_dd, image_aa, cell_mask, regions_mask)

        # 添加cell_id到统计结果
        stats['index'] = cell_id

        # 添加到总结果列表
        results.append(stats)

    if len(results) != 0:
        # 转换为DataFrame
        df = pd.DataFrame(results).set_index('index')
        return df
    else :
        return None


def calculate_cell_stats(image_dd, image_aa, cell_mask, regions_mask):
    """
    计算单个细胞内不同区域的强度比统计数据

    参数:
        image_dd: dd通道图像
        image_aa: aa通道图像
        cell_mask: 当前细胞的掩码
        regions_mask: 线粒体区域掩码

    返回:
        dict: 包含各项统计数据的字典
    """
    # 确保输入有效
    if np.sum(cell_mask) == 0:
        return {
            'Fp_mito_ratio_mean': np.nan,
            'Fp_non_mito_ratio_mean': np.nan,
            'Fp_whole_cell_ratio_mean': np.nan,
            'Fp_mito_pixels': 0,
            'Fp_non_mito_pixels': 0,
            'Fp_whole_cell_pixels': 0
        }

    # 计算线粒体区域（在当前细胞内的线粒体）
    mito_mask = cell_mask * regions_mask

    # 计算非线粒体区域
    non_mito_mask = cell_mask * (regions_mask == 0)

    # 提取各区域的像素值
    dd_mito = image_dd[mito_mask > 0]
    aa_mito = image_aa[mito_mask > 0]

    dd_non_mito = image_dd[non_mito_mask > 0]
    aa_non_mito = image_aa[non_mito_mask > 0]

    dd_whole = image_dd[cell_mask > 0]
    aa_whole = image_aa[cell_mask > 0]

    # 计算强度比（避免除以零）
    mito_ratio = calculate_ratio(aa_mito, dd_mito)
    non_mito_ratio = calculate_ratio(aa_non_mito, dd_non_mito)
    whole_ratio = calculate_ratio(aa_whole, dd_whole)

    # 返回统计结果
    return {
        'Fp_mito_ratio_mean': np.mean(mito_ratio) if len(mito_ratio) > 0 else np.nan,
        'Fp_non_mito_ratio_mean': np.mean(non_mito_ratio) if len(non_mito_ratio) > 0 else np.nan,
        'Fp_whole_cell_ratio_mean': np.mean(whole_ratio) if len(whole_ratio) > 0 else np.nan,
        'Fp_mito_pixels': len(mito_ratio),
        'Fp_non_mito_pixels': len(non_mito_ratio),
        'Fp_whole_cell_pixels': len(whole_ratio)
    }


def calculate_ratio(aa_values, dd_values):
    """
    计算AA/DD强度比，处理分母为零的情况

    参数:
        aa_values: AA通道像素值数组
        dd_values: DD通道像素值数组

    返回:
        np.array: 强度比数组
    """
    # 避免除以零
    valid_indices = dd_values > 0
    return aa_values[valid_indices] / dd_values[valid_indices]