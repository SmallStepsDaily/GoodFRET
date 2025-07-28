import os
import sys
import warnings
import cv2
import numpy as np
import tifffile as tiff

from extracting.phenotype.foxo3a_nuclei import calculate_fluorescence_ratio
from segmentation.seg import Segmentation, filter_labeled_masks_by_diameter
from cellpose import models

from tool.image import show_gray_image
from ui import Output

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")

class FOXO3ANucleiSegmentation(Segmentation):
    """
    基于
    """
    def __init__(self,
                 seg_diameter=200,
                 seg_min_diameter=100,
                 seg_max_diameter=600,
                 seg_nuclei_diameter=120,
                 seg_nuclei_min_diameter=80,
                 seg_nuclei_max_diameter=200,
                 output_redirector=Output()):
        super().__init__(output_redirector)
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.CellposeModel(gpu=True)
        self.seg_nuclei_diameter = seg_nuclei_diameter
        self.seg_nuclei_min_diameter = seg_nuclei_min_diameter
        self.seg_nuclei_max_diameter = seg_nuclei_max_diameter

    def start(self, path):
        # 读取细胞核和线粒体图像
        foxo3a_image_np = tiff.imread(os.path.join(path, 'Foxo3a.tif'))
        foxo3a_original_image_np = foxo3a_image_np.copy()
        nuclei_image_np = tiff.imread(os.path.join(path, 'Hoechst.tif'))
        # 将两个通道进行组合操作
        foxo3a_image_np = self.pretreatment(foxo3a_image_np)
        nuclei_image_np = self.pretreatment(nuclei_image_np)
        current_image_np = np.stack([foxo3a_image_np, nuclei_image_np], axis=-1)
        print(f"分割Foxo3a和细胞核组合的细胞操作 ===================>  {str(path)}")
        self.output.append(f"分割Foxo3a和细胞核组合的细胞操作 ===================>  {str(path)}")
        foxo3a_mask_np, nuclei_mask_np, original_nuclei_mask_np = self.segmentation(current_image_np)
        print(f"保存Foxo3a和细胞核组合的细胞操作 ===================>  {str(path)}")
        self.output.append(f"保存Foxo3a和细胞核组合的细胞操作 ===================>  {str(path)}")
        self.save(foxo3a_mask_np, path, 'Foxo3a_mask.tif')
        self.save(original_nuclei_mask_np, path, 'nmask.tif')

        # 开始计算对应的 foxo3a 的入核比例情况
        print("计算单细胞区域 foxo3a 入核情况 ===================> " + str(path))
        result = calculate_fluorescence_ratio(foxo3a_original_image_np, foxo3a_mask_np, nuclei_mask_np)
        result.to_csv(os.path.join(path, 'Foxo3a.csv'))
        print("保存单细胞区域 foxo3a 入核情况 ===================> " + str(path))
        return result

    def pretreatment(self, image_np):
        if image_np.mean() * 20 < image_np.max():
            print("该图像的存在较高的亮度，需要进行log对换降低最高亮度值！！！")
            result_image = self.log_image(image_np)
        else:
            result_image = image_np
        result_image = self.gaussian_image(result_image)
        return result_image


    def seg_foxo3a(self, image_np, factor):
        masks, flows, styles = self.seg_model.eval(image_np,
                                                          diameter=self.seg_diameter / factor,
                                                          flow_threshold=0.4)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)
        return masks_filtered

    def seg_nuclei(self, image_np, factor):
        masks, flows, styles = self.seg_model.eval(image_np,
                                                   diameter=self.seg_nuclei_diameter / factor,
                                                   flow_threshold=0.4)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_nuclei_min_diameter / factor,
                                                          max_diameter=self.seg_nuclei_max_diameter / factor)
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
        foxo3a_mask_np = self.seg_foxo3a(image_np[:, :, 0], factor)

        # 分割细胞核掩码返回图像
        original_nuclei_mask_np = self.seg_nuclei(image_np[:, :, 1], factor)

        foxo3a_mask_np, filter_nuclei_mask_np = self.common_mask(foxo3a_mask_np, original_nuclei_mask_np)

        if original_height == original_width:
            # 还原掩码到原始尺寸
            foxo3a_mask_np = cv2.resize(foxo3a_mask_np.astype(np.uint8), (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            filter_nuclei_mask_np = cv2.resize(filter_nuclei_mask_np.astype(np.uint8), (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            original_nuclei_mask_np = cv2.resize(original_nuclei_mask_np.astype(np.uint8),
                                               (original_width, original_height),
                                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return foxo3a_mask_np, filter_nuclei_mask_np, original_nuclei_mask_np


if __name__ == '__main__':
    """
    测试验证代码
    """
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ON!")
    else:
        print("CUDA is NOT available. GPU is OFF.")

    cell = FOXO3ANucleiSegmentation()
    cell.start(r'D:\data\qrm\2025.06.30 H1299 FOXO 0.5\H1299-vin-0.5h-d7-c110.40μm\7')