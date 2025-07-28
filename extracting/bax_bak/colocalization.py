import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu

from tool.image import show_gray_image


def count_single_cell_localization(image_dd, image_da, image_aa, mask, regions_mask):
    """
    统计单细胞内不同区域的AA/DD强度比、Manders重叠系数和Pearson相关系数

    参数:
        image_dd: dd通道图像
        image_da: da通道图像
        image_aa: aa通道图像
        mask: 细胞掩码，背景为0，每个细胞区域为唯一整数
        regions_mask: 线粒体区域掩码，背景为0，线粒体区域为1

    返回:
        df: 统计结果DataFrame，包含各细胞的强度比、重叠系数和相关系数
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
        cell_mask = np.where(mask == cell_id, 1, 0)

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
    else:
        return None


def calculate_cell_stats(image_dd, image_aa, cell_mask, regions_mask):
    """
    计算单个细胞内不同区域的强度比、Manders重叠系数和Pearson相关系数

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
            'Fp_region_ratio_mean': np.nan,
            'Fp_non_region_ratio_mean': np.nan,
            'Fp_cell_ratio_mean': np.nan,
            'Fp_region_pixels': 0,
            'Fp_non_region_pixels': 0,
            'Fp_cell_pixels': 0,
            # 新增指标
            'Fp_cell_MCC': np.nan,
            'Fp_region_PCC': np.nan,
            'Fp_cell_OCC': np.nan,
            'Fp_region_MOC': np.nan,
            'Fp_cell_MOC': np.nan,
        }

    # 计算线粒体区域（在当前细胞内的线粒体）
    mito_mask = cell_mask * regions_mask

    # 计算非线粒体区域
    non_mito_mask = cell_mask * (regions_mask == 0)

    # 提取单细胞区域的像素值
    cell_dd_image = image_dd[cell_mask > 0]

    cell_aa_image = image_aa[cell_mask > 0]

    dd_mito = image_dd[mito_mask > 0]
    aa_mito = image_aa[mito_mask > 0]

    if dd_mito is None:
        return {
            'Fp_region_ratio_mean': np.nan,
            'Fp_non_region_ratio_mean': np.nan,
            'Fp_cell_ratio_mean': np.nan,
            'Fp_region_pixels': 0,
            'Fp_non_region_pixels': 0,
            'Fp_cell_pixels': 0,
            # 新增指标
            'Fp_cell_MCC': np.nan,
            'Fp_region_PCC': np.nan,
            'Fp_cell_OCC': np.nan,
            'Fp_region_MOC': np.nan,
            'Fp_cell_MOC': np.nan,
        }

    dd_non_mito = image_dd[non_mito_mask > 0]
    aa_non_mito = image_aa[non_mito_mask > 0]

    dd_whole = image_dd[cell_mask > 0]
    aa_whole = image_aa[cell_mask > 0]

    # 计算强度比（避免除以零）
    mito_ratio = calculate_ratio(aa_mito, dd_mito)
    non_mito_ratio = calculate_ratio(aa_non_mito, dd_non_mito)
    whole_ratio = calculate_ratio(aa_whole, dd_whole)

    # 计算Manders重叠系数和Pearson相关系数
    manders_cell = calculate_manders(aa_whole, dd_whole)
    pearson_mito = calculate_pearson(aa_mito, dd_mito)
    pearson_cell = calculate_pearson(aa_whole, dd_whole)
    MOC_region = calculate_MOC(aa_mito, dd_mito)
    MOC_cell = calculate_MOC(aa_whole, dd_whole)

    # 返回统计结果（包含新增指标）
    return {
        'Fp_region_ratio_mean': np.mean(mito_ratio) if len(mito_ratio) > 0 else np.nan,
        'Fp_non_region_ratio_mean': np.mean(non_mito_ratio) if len(non_mito_ratio) > 0 else np.nan,
        'Fp_cell_ratio_mean': np.mean(whole_ratio) if len(whole_ratio) > 0 else np.nan,
        'Fp_region_pixels': len(mito_ratio),
        'Fp_non_region_pixels': len(non_mito_ratio),
        'Fp_cell_pixels': len(whole_ratio),
        # 新增指标
        'Fp_cell_MCC': manders_cell,
        'Fp_region_PCC': pearson_mito,
        'Fp_cell_PCC': pearson_cell,
        'Fp_region_MOC': MOC_region,
        'Fp_cell_MOC': MOC_cell,
    }


def calculate_ratio(aa_values, dd_values):
    """计算AA/DD强度比，处理分母为零的情况"""
    if dd_values is None:
        return np.nan
    valid_indices = dd_values > 0
    return aa_values[valid_indices] / dd_values[valid_indices]


def min_max_normalize(data, data_max, data_min, feature_range=(0, 1)):
    if len(data) == 0:
        return data
    """对输入数据进行Min-Max归一化"""
    if data_max == data_min:
        return np.full_like(data, feature_range[0])
    normalized_data = feature_range[0] + (data - data_min) * (feature_range[1] - feature_range[0]) / (data_max - data_min)
    return normalized_data

def calculate_MOC(channel1: np.array, channel2: np.array):
    """
    计算 AA 通道和 DD 通道中的MOC系数
    """
    if len(channel1) == 0:
        return np.nan
    return np.sum(channel1 * channel2) / np.sqrt((channel1 ** 2).sum() * (channel2 ** 2).sum())

def calculate_manders(channel1: np.array, channel2: np.array):
    """
    计算Manders重叠系数(Manders' overlap coefficient)

    参数:
        channel1: 第一个通道的像素值数组 为AA通道
        channel2: 第二个通道的像素值数组 为DD通道，

    返回:
        float: Manders重叠系数，值范围[0,1]，越接近1表示重叠越好
    """
    if len(channel1) < 10 or len(channel2) < 10:
        return np.nan

    # 使用Otsu方法计算阈值
    if len(np.unique(channel1)) < 2:  # 处理所有值相同的情况
        return 0.0

    channel1_threshold = threshold_otsu(channel1)
    channel2_threshold = threshold_otsu(channel2)
    mask1 = channel1 > channel1_threshold
    mask2 = channel2 > channel2_threshold

    # Manders重叠系数定义为channel2中与channel1重叠的部分占channel2总量的比例
    overlap = np.sum(channel2[mask1 & mask2]) / np.sum(channel2[mask2]) if np.sum(channel2[mask2]) > 0 else 0
    return min(max(overlap, 0), 1)  # 确保值在[0,1]范围内

def calculate_pearson(channel1: np.array, channel2: np.array):
    """
    计算Pearson相关系数
    
    参数:
        channel1: 第一个通道的像素值数组
        channel2: 第二个通道的像素值数组
        
    返回:
        float: Pearson相关系数，值范围[-1,1]，越接近1表示线性相关性越强
    """
    if len(channel1) < 10 or len(channel2) < 10:
        return np.nan
    
    try:
        # 使用scipy的pearsonr函数计算相关系数和p值
        corr, _ = pearsonr(channel1, channel2)
        return corr
    except Exception as e:
        print(f"Pearson相关系数计算错误: {e}")
        return np.nan