import os
import sys
import warnings
import cv2
import numpy as np
import tifffile as tiff
from segmentation.seg import Segmentation, filter_labeled_masks_by_diameter
from cellpose import models

from ui import Output

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")

class MitSegmentation(Segmentation):
    """
    线粒体通道组合图像进行分割
    """
    def __init__(self,
                 seg_diameter=200,
                 seg_min_diameter=100,
                 seg_max_diameter=600,
                 output_redirector=Output()):
        super().__init__(output_redirector)
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.CellposeModel(gpu=True)

    def start(self, path):
        # 读取细胞核和线粒体图像
        mit_image_np = tiff.imread(os.path.join(path, 'Mit.tif'))
        # 将两个通道进行组合操作
        mit_image_np = self.pretreatment(mit_image_np)
        print(f"分割线粒体掩码操作 ===================> {path}")
        self.output.append(f"分割线粒体掩码操作 ===================> {path}")
        mit_mask_np = self.segmentation(mit_image_np)
        print(f"保存线粒体掩码操作 ===================> {path}")
        self.output.append(f"保存线粒体掩码操作 ===================> {path}")
        self.save(mit_mask_np, path, 'mmask.tif')

    def pretreatment(self, image):
        result_image = self.gaussian_image(image)
        return result_image

    def seg_mit(self, image_np, factor):
        masks, flows, styles = self.seg_model.eval(image_np,
                                                   diameter=self.seg_diameter / factor,
                                                   flow_threshold=0.4)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)
        return masks_filtered

    def segmentation(self, image_np):
        """
        使用cellpose进行分割操作
        1. 下采样
        2. 分割
        :return:
        """
        # 记录原始图像尺寸
        original_height, original_width = image_np.shape[ : 2]
        factor = 1
        if original_height == original_width:
            # 下采样到 512x512
            image_np = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_AREA)
            factor = original_height / 512

        # 分割单细胞区域掩码返回图像
        mit_mask_np = self.seg_mit(image_np, factor)

        if original_height == original_width:
            # 还原掩码到原始尺寸
            mit_mask_np = cv2.resize(mit_mask_np.astype(np.uint8), (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return mit_mask_np


if __name__ == '__main__':
    """
    测试验证代码
    """
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ON!")
    else:
        print("CUDA is NOT available. GPU is OFF.")

    cell = MitSegmentation()
    cell.start(r'E:\data\qrm\2025.07.14\2025.07.01 A549 E-G 4H\A549-OSI-4h-d4-c11.75μm\11')