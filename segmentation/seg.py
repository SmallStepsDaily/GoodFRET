import os
import sys

import cv2
import numpy as np
import tifffile
from skimage import measure


class Segmentation:
    def __init__(self, output_redirector):
        self.seg_model = None
        self.seg_diameter = None
        self.seg_min_diameter = None
        self.seg_max_diameter = None
        self.output = output_redirector

    def start(self, path):
        """
        开始函数
        :return:
        """
        pass

    def pretreatment(self, image):
        """
        图像预处理
        :return:
        """
        pass

    def segmentation(self, image_np):
        """
        分割操作
        :return:
        """
        pass

    def save(self, image_np, output_path, image_name, dtype=np.uint8):
        """
        保存图像操作
        :return:
        """
        # 保存为图像文件
        tifffile.imwrite(str(os.path.join(output_path, image_name)), image_np.astype(dtype))


    @staticmethod
    def log_image(image_np):
        # 定义一个极小的正数，避免对数运算时出现趋近于 0 的情况
        epsilon = 1e-7
        # 确保图像数据中的每个元素都大于等于 epsilon
        image_np = np.maximum(image_np, epsilon)
        c = 255 / np.log(1 + np.max(image_np))
        log_transformed_image = c * (np.log(image_np + 1))
        log_transformed_image = log_transformed_image.astype(dtype=np.uint8)
        return log_transformed_image

    @staticmethod
    def gaussian_image(image_np):
        blurred_image = cv2.GaussianBlur(image_np, ksize=(9, 9), sigmaX=1.0)
        return blurred_image

    @staticmethod
    def common_mask(mit_mask_np, nuclei_mask_np):
        """
        提取线粒体和细胞核的对应区域，使用统一编号系统但保持各自独立
        """
        unique_mit_labels = np.unique(mit_mask_np)[1:]  # 排除背景标签0
        common_mit_mask = np.zeros_like(mit_mask_np)
        common_nuclei_mask = np.zeros_like(nuclei_mask_np)

        # 存储已匹配的细胞核标签，避免重复分配
        matched_nuclei = set()

        new_label = 1
        for mit_label in unique_mit_labels:
            mit_region = (mit_mask_np == mit_label)

            # 获取线粒体区域内的细胞核标签
            nuclei_in_mit = nuclei_mask_np[mit_region]
            nuclei_labels = np.unique(nuclei_in_mit[nuclei_in_mit > 0])

            if len(nuclei_labels) == 0:
                continue  # 无线粒体-细胞核重叠，跳过

            # 过滤已匹配的细胞核
            valid_nuclei = [n for n in nuclei_labels if n not in matched_nuclei]
            if not valid_nuclei:
                continue

            # 计算每个候选细胞核与当前线粒体的重叠面积
            overlap_areas = []
            for n_label in valid_nuclei:
                nuclei_region = (nuclei_mask_np == n_label)
                overlap = np.logical_and(mit_region, nuclei_region)
                overlap_areas.append((n_label, np.sum(overlap)))

            # 按重叠面积排序，选择最大的
            overlap_areas.sort(key=lambda x: x[1], reverse=True)
            main_nuclei_label, max_overlap = overlap_areas[0]

            # 计算双向覆盖率
            mit_coverage = max_overlap / np.sum(mit_region)
            nuclei_coverage = max_overlap / np.sum(nuclei_mask_np == main_nuclei_label)

            # 确保双向覆盖率都足够高
            if mit_coverage < 0.1 or nuclei_coverage < 0.1:  # 可调整阈值
                continue

            # 获取选中的细胞核区域
            nuclei_region = (nuclei_mask_np == main_nuclei_label)

            # 分配相同的编号给线粒体和细胞核区域
            common_mit_mask[mit_region] = new_label
            common_nuclei_mask[nuclei_region] = new_label

            # 标记该细胞核已被匹配
            matched_nuclei.add(main_nuclei_label)
            new_label += 1

        return common_mit_mask, common_nuclei_mask

    @staticmethod
    def show(image_np):
        """
                    输出图像为窗口形式
                    """
        if image_np is None:
            print("未提供图像数据。")
            return
        # 处理不同形状的 image_np
        if len(image_np.shape) == 3 and image_np.shape[0] == 1:
            # 如果是 1*2048*2048 的格式，去掉第一个维度
            image_np = np.squeeze(image_np, axis=0)

        # 创建可调整大小的窗口
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # 显示图像
        cv2.imshow('Image', image_np)
        # 等待按键事件
        cv2.waitKey(0)
        # 关闭所有窗口
        cv2.destroyAllWindows()


# 归一化图像到 [0, 255] 范围
def normalize_image(img):
    img_float32 = img.astype(np.float32)
    normalized_img = 255 * (img_float32 - img_float32.min()) / (img_float32.max() - img_float32.min())
    return normalized_img.astype(np.uint8)


def filter_labeled_masks_by_diameter(mask, min_diameter=80, max_diameter=150, border_distance=5):
    """
    过滤不符合条件细胞：
    1. 不符合规定直径
    2. 边缘细胞
    """
    # 获取所有标记区域的属性
    props = measure.regionprops(mask)
    # 创建一个空白掩码用于保存符合条件的细胞核
    filtered_mask = np.zeros_like(mask)
    # 新标签从1开始
    new_label = 1

    height, width = mask.shape

    for prop in props:
        # 检查等效圆直径是否在指定范围内
        if not (min_diameter <= prop.equivalent_diameter <= max_diameter):
            continue
        # 获取细胞的外接矩形
        min_row, min_col, max_row, max_col = prop.bbox

        # 检查是否为边缘细胞
        if min_row < border_distance or min_col < border_distance or \
                max_row > height - border_distance or max_col > width - border_distance:
            continue

        # 将符合条件的细胞核复制到新的掩码中
        filtered_mask[mask == prop.label] = new_label
        new_label += 1

    return filtered_mask
