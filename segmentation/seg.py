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
        # 命令行输出到文本框内
        self.original_stdout = sys.stdout
        sys.stdout = output_redirector

    def __del__(self):
        sys.stdout = self.original_stdout

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
    def show(self, image_np):
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


def filter_labeled_masks_by_diameter(mask, min_diameter=80, max_diameter=150):
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

    for prop in props:
        # 检查等效圆直径是否在指定范围内
        if not (min_diameter <= prop.equivalent_diameter <= max_diameter):
            continue
        # 获取当前细胞的掩码
        cell_mask = (mask == prop.label).astype(np.uint8)
        # 检测边缘
        edges = cv2.Canny(cell_mask, 50, 150)
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        # 如果检测到直线，认为是边缘细胞，跳过
        if lines is not None:
            continue
        # 将符合条件的细胞核复制到新的掩码中
        filtered_mask[mask == prop.label] = new_label
        new_label += 1

    return filtered_mask
