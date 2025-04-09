import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure
from skimage.feature import peak_local_max
from skimage.measure import regionprops


def count_single_cell_Ed(image, mask):
    """
    1. 计算单细胞的 Ed 效率值，并统计所有的效率分布情况
    """
    cell_averages = {}
    # 统计单个细胞中的 Ed 平均效率情况
    for cell_id in range(1, int(mask.max().item()) + 1):
        cell_mask = (mask == cell_id)
        cell_intensities = image[cell_mask]
        average_intensity = cell_intensities.mean().item()
        cell_averages[cell_id] = {}
        cell_averages[cell_id]['Ed_mean_value'] = average_intensity
        cell_averages[cell_id]['Ed_variance'] = (cell_intensities - average_intensity).pow(2).mean().item()

    # 创建一个 DataFrame
    result_df = pd.DataFrame.from_dict(cell_averages, orient='index')

    return result_df

def region_growing(image, seeds, threshold=30, max_points=100):
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

def region_growth_segmentation(original_img_tensor, mask_img_tensor):
    """
    在每个已标记的单细胞区域内进行聚点分割，并使用区域生长算法扩展聚点区域。

    算法存在问题后续进行优化操作

    :param original_img_tensor: 灰度图像 (numpy array)
    :param mask_img_tensor: 标记了单细胞区域的掩码图像 (numpy array)，其中不同细胞用不同的正整数标记
    :return: 分割后的二值图像，其中每个聚点及其扩展区域为1，其余位置为0
    """
    # 确保输入是numpy数组并转换为uint8类型
    image = np.array(original_img_tensor, dtype=np.float32)
    labeled_mask = np.array(mask_img_tensor, dtype=np.uint8)

    # 初始化输出二值图像
    binary_image = np.zeros_like(labeled_mask, dtype=np.uint8)

    # 获取所有细胞的属性
    regions = regionprops(labeled_mask)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cell_label = region.label
        cell_mask = (labeled_mask == cell_label)[minr:maxr, minc:maxc]
        cell_mask[cell_mask > 0] = 1
        cell_image = image[minr:maxr, minc:maxc] * cell_mask

        # 对于cell_image 进行归一化操作，方便计算
        cell_image_min = cell_image[cell_image > 0].min()
        cell_image_max = cell_image.max()

        # 单个区域在进行归一化操作 防止存在单细胞转换效率低的情况
        cell_image = 255 * (cell_image - cell_image_min) / (cell_image_max - cell_image_min)
        cell_image[cell_image < 0] = 0
        # 转换回无符号8位整数类型
        cell_image = cell_image.astype(np.uint8)

        # 预处理：高斯模糊减少噪声，增强对比度（可选）
        blurred = cv2.GaussianBlur(cell_image, (5, 5), 0)

        # 检测当前细胞内的局部最大值
        coordinates = peak_local_max(blurred, min_distance=20, threshold_abs=100)  # 调整参数以适应您的需求

        # 将坐标转换回原始图像坐标系
        coordinates[:, 0] += minr
        coordinates[:, 1] += minc

        # 创建标记图像
        markers = []
        for row, col in coordinates:
            if labeled_mask[row, col] == cell_label:  # 确保标记在该细胞区域内
                markers.append((row - minr, col - minc))

        # 应用区域生长算法
        grown_regions = region_growing(cell_image * cell_mask, markers, threshold=30)  # 调整阈值以适应您的需求
        # 连通区域分析
        filtered_grown_regions = filter_connected_components(grown_regions)
        # 更新全局二值图像
        binary_image[minr:maxr, minc:maxc][filtered_grown_regions != 0] = 1

    return binary_image

def filter_connected_components(segmented_image, min_size=20):
    """
    筛选连通组件，移除面积小于min_size或大于max_size的组件。

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


# def extracting_ed_features(image_mmask, image_nmask, image_DD, image_ED, image_RC):
#     """
#     提取 EGFR-GRB2 特征函数
#     """
#
#     pass