import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from tool.image import show_gray_image

"""
基于BAX靶点提取设计的FRET特征提取程序：
1. 针对AA通道进行聚点分割操作
"""


def min_max_normalize(image):
    """辅助函数：对非零区域进行Min-Max归一化"""
    mask = image > 0
    if np.sum(mask) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    min_val = np.min(image[mask])
    max_val = np.max(image[mask])
    if max_val == min_val:
        return (image * 255).astype(np.uint8)

    normalized = (image - min_val) / (max_val - min_val) * 255
    normalized = np.clip(normalized, 1, 255).astype(np.uint8)
    return normalized

def count_single_cell_Ed(image_ed, image_rc, image_dd, image_da, image_aa, background_noise_values, mask, rc_min=0.0, rc_max=0.5, ed_min=0.0, ed_max=1.0):
    """
    输入都是基于numpy处理的图像
    1. 计算单细胞的 Ed 效率值，并统计所有的效率分布情况
    2. 同时运行 region_growth_segmentation 代码，分析 image_dd 上的种子点，并将其应用到 image_ed 上
    """
    regions = regionprops(mask)

    # 拿到0为背景，细胞区域为1的掩码图像
    binary_mask = mask.copy()
    binary_mask[binary_mask > 0] = 1

    # 存放区域掩码图像
    regions_mask = np.zeros_like(mask, dtype=np.uint8)

    # 定义输出的结果
    result = {}

    for region in regions:
        # 细胞对应的标签
        cell_id = region.label

        # 细胞边框参数
        minr, minc, maxr, maxc = region.bbox

        # 对各类图像划分单细胞区域
        cell_mask = (mask == region.label)[minr:maxr, minc:maxc]
        cell_image_dd = image_dd[minr:maxr, minc:maxc] * cell_mask
        cell_image_aa = image_aa[minr:maxr, minc:maxc]
        cell_image_da = image_da[minr:maxr, minc:maxc]
        cell_image_ed = image_ed[minr:maxr, minc:maxc]
        cell_image_rc = image_rc[minr:maxr, minc:maxc]
        # 对于cell_image 进行uni8化操作，方便计算
        cell_image_normalize_dd = min_max_normalize(cell_image_dd)
        # show_gray_image(cell_image_normalize_dd)
        # 明亮区域增长
        region_mask = region_segmentation(cell_image_normalize_dd)

        # 筛选三通道背景阈值合理的位置点位
        # 判断是否aa通道满足大于三倍背景值的条件
        region_mask = filter_mask_by_intensity(region_mask, cell_image_aa, background_noise_values['AA'])
        # 判断是否da通道满足大于三倍背景值的条件
        region_mask = filter_mask_by_intensity(region_mask, cell_image_da, background_noise_values['DA'])

        # 筛选在符合Rc范围内的值
        # 创建RC图像的掩码（值在0到3之间的区域为True）
        rc_mask = (cell_image_rc >= rc_min) & (cell_image_rc <= rc_max)
        # 将原始mask与RC掩码相结合
        region_mask[~rc_mask] = 0  # 将RC值不在范围内的区域置为0

        # 筛选在符合Ed范围内的值
        # 创建RC图像的掩码（值在0到3之间的区域为True）
        ed_mask = (cell_image_ed >= ed_min) & (cell_image_ed <= ed_max)
        # 将原始mask与RC掩码相结合
        region_mask[~ed_mask] = 0  # 将RC值不在范围内的区域置为0

        # 筛选不符合区域大小的聚点
        region_mask = filter_connected_components(region_mask)

        # 将筛选得到的值放回原图像中
        regions_mask[minr:maxr, minc:maxc] = region_mask

        # 单细胞特征提取点 分别非0和存0两种图像进行采集
        cell_region_ed = cell_image_ed[cell_mask]
        cell_not_zero_average_ed = cell_region_ed[cell_region_ed > 0]
        result[cell_id] = {}
        if cell_region_ed.size > 0:
            result[cell_id]['Ed_mean_value'] = cell_region_ed.mean().item()
            result[cell_id]['Ed_variance'] = np.var(cell_region_ed).item()
        else:
            result[cell_id]['Ed_mean_value'] = 0
            result[cell_id]['Ed_variance'] = 0
        if cell_not_zero_average_ed.size > 0:
            result[cell_id]['Ed_not_zero_mean_value'] = cell_not_zero_average_ed.mean().item()
            result[cell_id]['Ed_not_zero_variance'] = np.var(cell_not_zero_average_ed).item()
        else:
            result[cell_id]['Ed_not_zero_mean_value'] = 0
            result[cell_id]['Ed_not_zero_variance'] = 0

        # 计算种子点的效率值
        region_ed = cell_image_ed[region_mask == 1]
        not_region_ed = cell_image_ed[(region_mask == 0) & (cell_mask == 1)]
        if region_ed.size > 0:
            result[cell_id]['Ed_region_mean_value'] = region_ed.mean().item()
            result[cell_id]['Ed_region_variance'] = np.var(region_ed).item()
            result[cell_id]['Ed_region_max_value'] = region_ed.max().item()
            result[cell_id]['Ed_region_min_value'] = region_ed.min().item()
            result[cell_id]['Ed_region_top_50_value'] = top_50_percent_average(region_ed)
            result[cell_id]['Ed_region_top_25_value'] = top_25_percent_average(region_ed)
        else:
            result[cell_id]['Ed_region_mean_value'] = 0
            result[cell_id]['Ed_region_variance'] = 0
            result[cell_id]['Ed_region_max_value'] = 0
            result[cell_id]['Ed_region_min_value'] = 0
            result[cell_id]['Ed_region_top_50_value'] = 0
            result[cell_id]['Ed_region_top_25_value'] = 0
        if not_region_ed.size > 0:
            result[cell_id]['Ed_not_region_mean_value'] = not_region_ed.mean().item()
            result[cell_id]['Ed_not_region_variance'] = np.var(not_region_ed).item()
            result[cell_id]['Ed_not_region_max_value'] = not_region_ed.max().item()
            result[cell_id]['Ed_not_region_min_value'] = not_region_ed.min().item()
            result[cell_id]['Ed_not_region_top_50_value'] = top_50_percent_average(not_region_ed)
            result[cell_id]['Ed_not_region_top_25_value'] = top_25_percent_average(not_region_ed)
        else:
            result[cell_id]['Ed_not_region_mean_value'] = 0
            result[cell_id]['Ed_not_region_variance'] = 0
            result[cell_id]['Ed_not_region_max_value'] = 0
            result[cell_id]['Ed_not_region_min_value'] = 0
            result[cell_id]['Ed_not_region_top_50_value'] = 0
            result[cell_id]['Ed_not_region_top_25_value'] = 0
        print(str(result[cell_id]['Ed_region_variance']), "均值效率: ", result[cell_id]['Ed_region_mean_value'], " 前百分之50的效率值: ", result[cell_id]['Ed_not_region_top_50_value'])
    # 创建一个 DataFrame
    result_df = pd.DataFrame.from_dict(result, orient='index')
    return result_df, regions_mask


def top_25_percent_average(arr):
    """
    计算数组前百分之25的值的平均值

    :param arr: 输入的NumPy数组
    :return: 数组前百分之25的值的平均值
    """
    # 对数组进行降序排序
    sorted_arr = np.sort(arr)[::-1]
    # 计算前25%的索引位置
    index = int(len(arr) * 0.25)
    # 取前50%的元素
    top_25_elements = sorted_arr[:index]
    # 计算这些元素的平均值
    average = np.mean(top_25_elements)
    return average


def top_50_percent_average(arr):
    """
    计算数组由高到低前百分之50的值的平均值

    :param arr: 输入的NumPy数组
    :return: 数组由高到低前百分之50的值的平均值
    """
    # 对数组进行降序排序
    sorted_arr = np.sort(arr)[::-1]
    # 计算前50%的索引位置
    index = int(len(arr) * 0.5)
    # 取前50%的元素
    top_50_elements = sorted_arr[:index]
    # 计算这些元素的平均值
    average = np.mean(top_50_elements)
    return average


def filter_mask_by_intensity(seeds_mask, image_aa, aa_value, background_factor=2.0):
    """
    根据强度值筛选掩码区域，生成新的掩码图像

    参数:
        seeds_mask: 原始掩码图像 (numpy数组，二值图像，非零值表示掩码区域)
        image_aa: 用于筛选的强度图像 (numpy数组，与seeds_mask尺寸相同)
        aa_value: 强度阈值系数，筛选条件为 强度 > 3*aa_value

    返回:
        filtered_mask: 筛选后的新掩码图像 (二值图像，满足条件的区域为1，其余为0)
    """
    # 确保输入图像尺寸一致
    if seeds_mask.shape != image_aa.shape:
        raise ValueError("seeds_mask和image_aa的尺寸必须一致")

    # 创建新掩码，初始化为全0
    filtered_mask = np.zeros_like(seeds_mask, dtype=np.uint8)

    # 在原始掩码区域内筛选强度大于3*aa_value的像素
    mask_region = seeds_mask > 0
    valid_pixels = (image_aa > background_factor * aa_value) & mask_region

    # 将满足条件的像素在新掩码中设为1
    filtered_mask[valid_pixels] = 1

    return filtered_mask


def region_segmentation(image_dd):
    """
    分割单细胞图像中的丝状/片状亮区域

    参数:
        image_dd: 单细胞框选图，背景为0，细胞区域>0

    返回:
        bright_regions_mask: 分割出的亮区域掩码
    """
    # 创建细胞掩码（去除背景）
    cell_mask = (image_dd > 0).astype(np.uint8)

    # 如果没有细胞区域，直接返回空掩码
    if np.sum(cell_mask) == 0:
        return np.zeros_like(image_dd, dtype=np.uint8)

    # 仅保留细胞区域的图像
    cell_image = image_dd * cell_mask

    # 仅在细胞区域内计算Otsu阈值
    cell_pixels = cell_image[cell_image > cell_image.mean()]

    if cell_pixels.size == 0:  # 防止所有细胞像素都被滤波为0
        return np.zeros_like(image_dd, dtype=np.uint8)

    # 计算细胞区域的Otsu阈值
    otsu_threshold, _ = cv2.threshold(
        cell_pixels, 155, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 应用阈值到整个图像
    global_mask = np.zeros_like(cell_image, dtype=np.uint8)
    global_mask[cell_image >= otsu_threshold] = 1

    return global_mask


def filter_connected_components(segmented_image, min_size=10):
    """
    筛选连通组件，移除面积小于min_size的区域。

    :param segmented_image: 区域生长后的二值图像
    :param min_size: 最小连通区域面积，默认20
    :return: 过滤后的二值图像
    """
    # 使用 OpenCV 找到所有连通组件及其统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

    filtered_image = np.zeros_like(segmented_image, dtype=np.uint8)
    # 创建一个掩码来保存符合条件的连通区域
    mask = np.zeros_like(segmented_image, dtype=bool)

    # 如果细胞存在
    for i in range(1, num_labels):  # 跳过背景（标签0）
        if min_size <= stats[i, cv2.CC_STAT_AREA]:
            mask[labels == i] = True
    # 应用掩码生成最终结果
    filtered_image[mask] = 1
    return filtered_image