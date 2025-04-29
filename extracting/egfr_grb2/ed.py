import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from skimage.feature import peak_local_max
from skimage.measure import regionprops

def min_max_normalize(image, target_min=0, target_max=255):
    """
    对输入图像进行Min-Max归一化

    :param image: 输入的图像数组，形状为 (C, H, W) 或 (H, W)
    :param target_min: 目标范围的最小值，默认为0
    :param target_max: 目标范围的最大值，默认为255
    :return: 归一化后的图像数组，类型为uint8
    """
    min_val = np.min(image)
    max_val = np.max(image)
    denominator = max_val - min_val
    denominator = np.where(denominator == 0, 1e-8, denominator)
    normalized_image = (image - min_val) / denominator
    normalized_image = target_min + (target_max - target_min) * normalized_image
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
    return normalized_image

def count_single_cell_Ed(image_ed, image_rc, image_dd, mask, rc_min=0.0, rc_max=2.5, ed_min=0.0, ed_max=1.0):
    """
    输入都是基于numpy处理的图像
    1. 计算单细胞的 Ed 效率值，并统计所有的效率分布情况
    2. 同时运行 region_growth_segmentation 代码，分析 image_dd 上的种子点，并将其应用到 image_ed 上
    """
    regions = regionprops(mask)

    # 在这里要定义rc的值，当rc的值在 rc_min-rc_max 之间表示合理区间
    rc_mask = image_rc.copy()
    rc_mask[rc_mask > rc_max] = 0
    rc_mask[rc_mask < rc_min] = 0
    rc_mask[rc_mask > 0] = 1

    # 拿到0为背景，细胞区域为1的掩码图像
    binary_mask = mask.copy()
    binary_mask[binary_mask > 0] = 1

    # 掩码不合理的效率特征
    rc_mask = rc_mask * binary_mask

    # 在这里要筛选ed的值，当ed的值在 ed_min-ed_max 之间表示合理区间
    image_ed[image_ed > ed_max] = 0
    image_ed[image_ed < ed_min] = 0

    # 掩码对应的点以判断增长点
    image_dd = image_dd * rc_mask
    image_ed = image_ed * rc_mask

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
        cell_image_ed = image_ed[minr:maxr, minc:maxc]

        # 对于cell_image 进行归一化操作，方便计算
        cell_image_normalize_dd = min_max_normalize(cell_image_dd)

        # 种子点增长
        seeds_mask = region_growth_segmentation(cell_image_normalize_dd)

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
        seed_region_ed = cell_image_ed[seeds_mask == 1]
        if seed_region_ed.size > 0:
            result[cell_id]['Ed_agg_mean_value'] = seed_region_ed.mean().item()
            result[cell_id]['Ed_agg_variance'] = np.var(seed_region_ed).item()
            result[cell_id]['Ed_agg_max_value'] = seed_region_ed.max().item()
            result[cell_id]['Ed_agg_min_value'] = seed_region_ed.min().item()
            result[cell_id]['Ed_agg_top_50_value'] = top_50_percent_average(seed_region_ed)
            result[cell_id]['Ed_agg_top_25_value'] = top_25_percent_average(seed_region_ed)
        else:
            result[cell_id]['Ed_agg_mean_value'] = 0
            result[cell_id]['Ed_agg_variance'] = 0
            result[cell_id]['Ed_agg_max_value'] = 0
            result[cell_id]['Ed_agg_min_value'] = 0
            result[cell_id]['Ed_agg_top_50_value'] = 0
            result[cell_id]['Ed_agg_top_25_value'] = 0
    # 创建一个 DataFrame
    result_df = pd.DataFrame.from_dict(result, orient='index')
    return result_df

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



def region_growth_segmentation(image):
    """
    在图像内进行聚点分割，并使用区域生长算法扩展聚点区域。

    :param image: 灰度图像 (numpy array)
    :return: 分割后的二值图像，其中每个聚点及其扩展区域为 1 ，其余位置为0
    """

    # 预处理：高斯模糊减少噪声，增强对比度（可选）
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # 检测当前图像内的局部最大值作为种子点
    coordinates = peak_local_max(blurred, min_distance=20, threshold_abs=100)  # 调整参数以适应您的需求

    # 存在值都很高的情况下，随机采集点进行分析
    if len(coordinates) == 0:
        coordinates = select_top_points(blurred, num_points=5, min_distance=20)

    # 创建标记图像（种子点列表）
    seeds = [tuple(coord) for coord in coordinates]

    # 应用区域生长算法
    grown_regions = region_growth(image, seeds, threshold=30)  # 调整阈值以适应您的需求

    # 连通区域分析
    filtered_grown_regions = filter_connected_components(grown_regions)

    return filtered_grown_regions

def select_top_points(image, num_points=5, min_distance=20):
    """
    选择图像中灰度值最高的 num_points 个点，且点之间的距离至少为 min_distance 像素

    :param image: 输入图像 (numpy array)
    :param num_points: 要选择的点的数量
    :param min_distance: 点之间的最小距离
    :return: 选择的点的列表 [(row, col)]
    """
    flat_image = image.flatten()
    sorted_indices = np.argsort(flat_image)[::-1]
    rows, cols = image.shape
    selected_points = []

    for index in sorted_indices:
        point = (index // cols, index % cols)
        valid = True
        for selected_point in selected_points:
            distance = np.sqrt((point[0] - selected_point[0]) ** 2 + (point[1] - selected_point[1]) ** 2)
            if distance < min_distance:
                valid = False
                break
        if valid:
            selected_points.append(point)
            if len(selected_points) == num_points:
                break

    return selected_points

def region_growth(image, seeds, threshold=30, max_points=100):
    """
    执行区域生长算法，并限制最大生长点数。

    :param image: 输入图像 (numpy array)
    :param seeds: 种子点列表 [(row, col)]
    :param threshold: 生长阈值，表示允许的最大灰度差异
    :param max_points: 最大生长点数，默认为100
    :return: 区域生长后的二值图像
    """
    segmented = np.zeros_like(image, dtype=np.uint8)
    processed = np.zeros_like(image, dtype=bool)
    s = generate_binary_structure(2, 2)  # 8-连通结构

    seed_values = {seed: image[seed] for seed in seeds}  # 记录每个种子点的灰度值

    for seed in seeds:
        queue = [seed]
        processed[seed[0], seed[1]] = True
        segmented[seed[0], seed[1]] = 1
        point_count = 1
        while queue:
            current = queue.pop(0)
            current_seed_value = seed_values[seed]  # 使用种子点的灰度值作为参考

            for offset in zip(*s.nonzero()):
                x_new, y_new = current[0] + offset[0] - 1, current[1] + offset[1] - 1
                if (0 <= x_new < image.shape[0]) and (0 <= y_new < image.shape[1]):
                    neighbor_value = image[x_new, y_new]
                    if abs(int(neighbor_value) - int(current_seed_value)) < threshold:
                        if not processed[x_new, y_new]:
                            segmented[x_new, y_new] = 1
                            queue.append((x_new, y_new))
                            processed[x_new, y_new] = True
                        point_count += 1  # 更新计数器
                if point_count >= max_points:  # 达到最大点数后停止生长
                    break
    return segmented

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