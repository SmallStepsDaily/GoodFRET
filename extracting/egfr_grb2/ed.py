import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import generate_binary_structure
from skimage.feature import peak_local_max
from skimage.measure import regionprops

def min_max_normalize(image, target_min=0, target_max=255):
    """
    仅对单细胞区域（像素值>0）进行Min-Max归一化，保持背景（像素值=0）不变

    :param image: 输入的图像数组，形状为 (C, H, W) 或 (H, W)
    :param target_min: 目标范围的最小值，默认为0
    :param target_max: 目标范围的最大值，默认为255
    :return: 归一化后的图像数组，类型为uint8
    """
    # 创建输出数组的副本，保留原始背景
    normalized_image = image.copy()

    # 仅处理单细胞区域（像素值>0）
    cell_mask = image > 0

    if np.any(cell_mask):  # 检查是否存在单细胞区域
        # 计算单细胞区域的最小值和最大值
        cell_values = image[cell_mask]
        min_val = np.min(cell_values)
        max_val = np.max(cell_values)
        # 避免除零错误
        denominator = max_val - min_val
        if denominator == 0:
            denominator = 1e-8  # 防止除以零

        # 对单细胞区域进行归一化
        cell_normalized = (cell_values - min_val) / denominator
        cell_normalized = target_min + (target_max - target_min) * cell_normalized
        cell_normalized = np.clip(cell_normalized, target_min, target_max)

        # 将归一化后的单细胞区域放回输出图像
        normalized_image[cell_mask] = cell_normalized.astype(np.uint8)

    return normalized_image

def count_single_cell_Ed(image_ed,
                         image_rc,
                         image_dd,
                         image_da,
                         image_aa,
                         background_noise_values,
                         mask,
                         rc_min=0.0,
                         rc_max=3,
                         ed_min=0.0,
                         ed_max=1.0):
    """
    输入都是基于numpy处理的图像
    1. 计算单细胞的 Ed 效率值，并统计所有的效率分布情况
    2. 同时运行 region_growth_segmentation 代码，分析 image_dd 上的种子点，并将其应用到 image_ed 上
    """
    # print("掩码最大值", mask.max())
    regions = regionprops(mask)

    # 获取聚点掩码图像进行保存
    agg_mask = np.zeros_like(mask, dtype=np.uint8)

    # 定义输出的结果
    result = {}
    # print("单细胞区域数量", len(regions))
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

        # 确定增长阈值范围
        valid_cell_image_dd = cell_image_normalize_dd[cell_image_normalize_dd > 0]
        # 计算前10%的阈值（第90百分位数）
        threshold = np.percentile(valid_cell_image_dd, 90)
        # 单细胞背景均值
        cell_background_value = valid_cell_image_dd.mean()
        # 提取前5%的元素
        top_percent = valid_cell_image_dd[valid_cell_image_dd > threshold].mean()
        # print(top_percent, cell_background_value)
        # 种子点增长
        seeds_mask = region_growth_segmentation(cell_image_normalize_dd,
                                                threshold=(top_percent - cell_background_value),
                                                threshold_abs=top_percent)

        # 筛选三通道背景阈值合理的位置点位
        # 判断对应的聚点是否aa通道满足大于2倍背景值的条件
        seeds_mask = filter_mask_by_intensity(seeds_mask, cell_image_aa, background_noise_values['AA'])
        # 判断对应的聚点是否da通道满足大于2倍背景值的条件
        seeds_mask = filter_mask_by_intensity(seeds_mask, cell_image_da, background_noise_values['DA'])


        # 筛选在符合Rc范围内的值
        # 创建RC图像的掩码（值在0到3之间的区域为True）
        rc_mask = (cell_image_rc >= rc_min) & (cell_image_rc <= rc_max)
        # 将原始mask与RC掩码相结合
        seeds_mask[~rc_mask] = 0  # 将RC值不在范围内的区域置为0

        # 筛选在符合Ed范围内的值
        # 创建RC图像的掩码（值在0到3之间的区域为True）
        ed_mask = (cell_image_ed >= ed_min) & (cell_image_ed <= ed_max)
        # 将原始mask与RC掩码相结合
        seeds_mask[~ed_mask] = 0  # 将RC值不在范围内的区域置为0

        # 筛选不符合区域大小的聚点
        seeds_mask = filter_connected_components(seeds_mask)

        # 获取前50的聚点进行分析
        seeds_mask = get_top_intensity_regions(seeds_mask, cell_image_normalize_dd, 50)

        # 保存合格的掩码
        agg_mask[minr:maxr, minc:maxc] = seeds_mask | agg_mask[minr:maxr, minc:maxc]

        # 单细胞特征提取点 分别非0和存0两种图像进行采集
        cell_region_ed = cell_image_ed[cell_mask]
        cell_not_zero_average_ed = cell_region_ed[cell_region_ed > 0]
        result[cell_id] = {}
        if cell_region_ed.size > 0:
            result[cell_id]['mean'] = cell_region_ed.mean().item()
            result[cell_id]['variance'] = np.var(cell_region_ed).item()
        else:
            result[cell_id]['mean'] = np.nan
            result[cell_id]['variance'] = np.nan
        if cell_not_zero_average_ed.size > 0:
            result[cell_id]['not_zero_mean'] = cell_not_zero_average_ed.mean().item()
            result[cell_id]['not_zero_variance'] = np.var(cell_not_zero_average_ed).item()
        else:
            result[cell_id]['not_zero_mean'] = np.nan
            result[cell_id]['not_zero_variance'] = np.nan

        # 计算种子点的效率值
        seed_region_ed = cell_image_ed[seeds_mask == 1]
        if seed_region_ed.size > 0:
            result[cell_id]['region_mean'] = seed_region_ed.mean().item()
            result[cell_id]['region_rc_mean'] = cell_image_rc[seeds_mask == 1].mean().item()
            result[cell_id]['region_variance'] = np.var(seed_region_ed).item()
            result[cell_id]['region_max'] = seed_region_ed.max().item()
            result[cell_id]['region_min'] = seed_region_ed.min().item()
            result[cell_id]['region_top_50'] = top_50_percent_average(seed_region_ed)
            result[cell_id]['region_top_25'] = top_25_percent_average(seed_region_ed)
        else:
            result[cell_id]['region_mean'] = np.nan
            result[cell_id]['region_rc_mean'] = np.nan
            result[cell_id]['region_variance'] = np.nan
            result[cell_id]['region_max'] = np.nan
            result[cell_id]['region_min'] = np.nan
            result[cell_id]['region_top_50'] = np.nan
            result[cell_id]['region_top_25'] = np.nan
        # 验证输出的ed是否正确
        # print(result[cell_id]['Ed_agg_top_50_value'])
    # 创建一个 DataFrame
    result_df = pd.DataFrame.from_dict(result, orient='index')
    return result_df, agg_mask


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

    # 在原始掩码区域内筛选强度大于 background_factor * aa_value 的像素
    mask_region = seeds_mask > 0
    valid_pixels = (image_aa > background_factor * aa_value) & mask_region

    # 将满足条件的像素在新掩码中设为1
    filtered_mask[valid_pixels] = 1

    return filtered_mask

def region_growth_segmentation(image_dd, threshold=15, threshold_abs=125):
    """
    在图像内进行聚点分割，并使用区域生长算法扩展聚点区域。

    :param image_dd: 灰度图像 (numpy array)
    :param threshold: 区域增长强度范围
    :param threshold_abs: 种子点阈值范围
    :return: 分割后的二值图像，其中每个聚点及其扩展区域为 1 ，其余位置为0
    """

    # 预处理：高斯模糊减少噪声，增强对比度（可选）
    blurred = cv2.GaussianBlur(image_dd, (3, 3), 0)

    # 检测当前图像内的局部最大值作为种子点
    coordinates = peak_local_max(blurred, min_distance=20, threshold_abs=threshold_abs)  # 调整参数以适应您的需求

    # 创建标记图像（种子点列表）
    seeds = [tuple(coord) for coord in coordinates]

    # 应用区域生长算法
    grown_regions = region_growth(image_dd, seeds, threshold=threshold)  # 调整阈值以适应您的需求
    # show_gray_image(grown_regions * 255)
    # 连通区域分析
    filtered_grown_regions = filter_connected_components(grown_regions)

    return filtered_grown_regions

def region_growth(image, seeds, threshold=20, max_points=400):
    """
    执行区域生长算法，并按每个种子点独立限制最大生长点数。

    :param image: 输入图像 (numpy array)
    :param seeds: 种子点列表 [(row, col)]
    :param threshold: 生长阈值，表示允许的最大灰度差异
    :param max_points: 每个种子点允许的最大生长点数，默认为100
    :return: 区域生长后的二值图像
    """
    if len(seeds) == 0:
        return np.zeros_like(image, dtype=np.uint8)

    segmented = np.zeros_like(image, dtype=np.uint8)
    processed = np.zeros_like(image, dtype=bool)
    s = generate_binary_structure(2, 2)  # 8-连通结构

    # 记录每个种子点的灰度值
    seed_values = {seed: image[seed] for seed in seeds}

    for seed in seeds:
        # 跳过已经处理过的种子点
        if processed[seed]:
            continue

        queue = [seed]
        processed[seed] = True
        segmented[seed] = 1
        point_count = 1  # 每个种子点独立计数

        while queue and point_count < max_points:
            current = queue.pop(0)
            current_seed_value = seed_values[seed]

            # 遍历8-连通邻域
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    if not s[i, j]:  # 跳过结构元素中为0的位置
                        continue

                    # 计算邻域点坐标
                    x_new = current[0] + i - 1  # 减1是为了将中心移到当前点
                    y_new = current[1] + j - 1

                    # 检查边界
                    if 0 <= x_new < image.shape[0] and 0 <= y_new < image.shape[1]:
                        # 检查是否已处理
                        if not processed[x_new, y_new]:
                            # 检查灰度差异
                            if abs(int(image[x_new, y_new]) - int(current_seed_value)) < threshold:
                                # 检查点数限制
                                if point_count < max_points:
                                    segmented[x_new, y_new] = 1
                                    queue.append((x_new, y_new))
                                    processed[x_new, y_new] = True
                                    point_count += 1
                                else:
                                    break  # 跳出内层for循环
                # 如果达到点数限制，跳出外层for循环
                if point_count >= max_points:
                    break

    return segmented

def get_top_intensity_regions(seeds_mask, image_dd, n=30):
    """
    获取掩码中荧光强度均值最高的前N个连通区域，若不足N个则返回原掩码

    参数:
        seeds_mask: 原始掩码图像 (numpy数组，二值图像，非零值表示掩码区域)
        image_dd: 荧光强度图像 (numpy数组，与seeds_mask尺寸相同)
        n: 需要保留的最高强度区域数量，默认为5

    返回:
        若连通区域数量≥n: 仅包含前N个高强度区域的新掩码图像
        若连通区域数量<n: 返回原始掩码seeds_mask
    """
    # 确保输入图像尺寸一致
    if seeds_mask.shape != image_dd.shape:
        raise ValueError("seeds_mask和image_dd的尺寸必须一致")

    # 标记连通区域
    labeled_array, num_features = ndimage.label(seeds_mask)

    # 如果连通区域数量不足n，直接返回原掩码
    if num_features < n:
        return seeds_mask

    # 计算每个连通区域的荧光强度均值
    region_means = ndimage.mean(image_dd, labeled_array, index=range(1, num_features + 1))

    # 获取强度均值最高的前N个区域的索引
    top_indices = np.argsort(-np.array(region_means))[:n] + 1  # +1是因为region index从1开始

    # 创建只包含前N个区域的新掩码
    top_regions_mask = np.isin(labeled_array, top_indices).astype(np.uint8)

    return top_regions_mask

def filter_connected_components(segmented_image, min_size=50):
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