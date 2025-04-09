import os
import warnings
import cv2
import numpy as np
import tifffile as tiff
from segmentation.seg import Segmentation, filter_labeled_masks_by_diameter
from cellpose import models

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")

class MitNucleiSegmentation(Segmentation):
    """
    线粒体和nuclei两通道组合图像进行分割
    """
    def __init__(self,
                 seg_diameter=200,
                 seg_min_diameter=100,
                 seg_max_diameter=600,
                 seg_nuclei_diameter=120,
                 seg_nuclei_min_diameter=80,
                 seg_nuclei_max_diameter=200,
                 output_redirector=None):
        super().__init__(output_redirector)
        self.channel = [1, 2]
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.Cellpose(gpu=True, model_type='cyto3')
        self.seg_nuclei_diameter = seg_nuclei_diameter
        self.seg_nuclei_min_diameter = seg_nuclei_min_diameter
        self.seg_nuclei_max_diameter = seg_nuclei_max_diameter
        self.seg_nuclei_model = models.Cellpose(gpu=True, model_type='nuclei')

    def start(self, path):
        # 读取细胞核和线粒体图像
        mit_image_np = tiff.imread(os.path.join(path, 'Mit.tif'))
        nuclei_image_np = tiff.imread(os.path.join(path, 'Hoechst.tif'))
        # 将两个通道进行组合操作
        mit_image_np = self.pretreatment(mit_image_np)
        nuclei_image_np = self.pretreatment(nuclei_image_np)
        current_image_np = np.stack([mit_image_np, nuclei_image_np], axis=-1)
        print("分割线粒体和细胞核组合的细胞操作 ===================> " + str(path))
        mit_mask_np, nuclei_mask_np = self.segmentation(current_image_np)
        print("保存线粒体和细胞核组合的细胞mask ===================> " + str(path))
        self.save(mit_mask_np, path, 'mmask.jpg')
        self.save(nuclei_mask_np, path, 'nmask.jpg')

    def pretreatment(self, image):
        result_image = self.log_image(image)
        result_image = self.gaussian_image(result_image)
        return result_image


    def seg_mit_nuclei(self, image_np, factor):
        masks, flows, styles, diams = self.seg_model.eval(image_np,
                                                          diameter=self.seg_diameter / factor,
                                                          channels=self.channel,
                                                          flow_threshold=0.4,
                                                          resample=True,
                                                          do_3D=False)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)
        return masks_filtered

    def seg_nuclei(self, image_np, factor):
        masks, flows, styles, diams = self.seg_nuclei_model.eval(image_np,
                                                                    diameter=self.seg_nuclei_diameter / factor,
                                                                    channels=[0, 0],
                                                                    flow_threshold=0.4,
                                                                    resample=True,
                                                                    do_3D=False)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_nuclei_min_diameter / factor,
                                                          max_diameter=self.seg_nuclei_max_diameter / factor)
        return masks_filtered

    @staticmethod
    def common_mask(mit_mask_np, nuclei_mask_np):
        """
        提取共同区域
        """
        unique_mit_labels = np.unique(mit_mask_np)[1:]  # 排除背景标签 0
        unique_nuclei_labels = np.unique(nuclei_mask_np)[1:]  # 排除背景标签 0

        common_mit_mask = np.zeros_like(mit_mask_np)
        common_nuclei_mask = np.zeros_like(nuclei_mask_np)

        new_label = 1
        for mit_label in unique_mit_labels:
            mit_region = (mit_mask_np == mit_label)
            # 检查线粒体区域内是否有细胞核
            nuclei_in_mit = nuclei_mask_np[mit_region]
            if np.any(nuclei_in_mit):
                # 找到该线粒体区域内的主要细胞核标签
                main_nuclei_label = np.bincount(nuclei_in_mit[nuclei_in_mit > 0]).argmax()
                nuclei_region = (nuclei_mask_np == main_nuclei_label)
                # 检查该细胞核区域内是否有线粒体
                mit_in_nuclei = mit_mask_np[nuclei_region]
                if np.any(mit_in_nuclei):
                    # 标记共同区域
                    common_mit_mask[mit_region] = new_label
                    common_nuclei_mask[nuclei_region] = new_label
                    new_label += 1

        return common_mit_mask, common_nuclei_mask

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
        mit_mask_np = self.seg_mit_nuclei(image_np, factor)
        # 分割细胞核掩码返回图像
        nuclei_mask_np = self.seg_nuclei(image_np[:, :, 1], factor)

        mit_mask_np, nuclei_mask_np = self.common_mask(mit_mask_np, nuclei_mask_np)

        if original_height == original_width:
            # 还原掩码到原始尺寸
            mit_mask_np = cv2.resize(mit_mask_np.astype(np.uint8), (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            nuclei_mask_np = cv2.resize(nuclei_mask_np.astype(np.uint8), (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return mit_mask_np, nuclei_mask_np


if __name__ == '__main__':
    """
    测试验证代码
    """
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ON!")
    else:
        print("CUDA is NOT available. GPU is OFF.")

    cell = MitNucleiSegmentation()
    cell.start(r'D:\data\qrm\2025.03.26 A549 24H\ALM\6')