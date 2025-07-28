import os
import sys
import warnings
import cv2
import numpy as np
import tifffile as tiff
from segmentation.seg import Segmentation, filter_labeled_masks_by_diameter, normalize_image
from cellpose import models

from tool.image import show_gray_image
from ui import Output

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
                 output_redirector=Output()):
        super().__init__(output_redirector)
        self.original_width = 2048
        self.original_height = 2048
        self.factor = 1
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.CellposeModel(gpu=True)
        self.seg_nuclei_diameter = seg_nuclei_diameter
        self.seg_nuclei_min_diameter = seg_nuclei_min_diameter
        self.seg_nuclei_max_diameter = seg_nuclei_max_diameter

    def start(self, path):
        try:
            # 读取细胞核和线粒体图像
            mit_image_np = tiff.imread(os.path.join(path, 'Mit.tif'))
            nuclei_image_np = tiff.imread(os.path.join(path, 'Hoechst.tif'))
            # 记录原始图像尺寸
            self.original_height, self.original_width = mit_image_np.shape
            self.factor = self.original_height / 512
            # 将两个通道进行组合操作
            mit_image_np = self.pretreatment(mit_image_np)
            nuclei_image_np = self.pretreatment(nuclei_image_np, need_CLAHE=True)
            current_image_np = np.stack([mit_image_np, nuclei_image_np], axis=-1)
            print(f"分割线粒体和细胞核组合的细胞操作 ===================>  {str(path)}")
            self.output.append(f"分割线粒体和细胞核组合的细胞操作 ===================>  {str(path)}")
            mit_mask_np, nuclei_mask_np = self.segmentation(current_image_np)
            print(f"保存线粒体和细胞核组合的细胞操作 ===================>  {str(path)}")
            self.output.append(f"保存线粒体和细胞核组合的细胞操作 ===================>  {str(path)}")
            self.save(mit_mask_np, path, 'mmask.tif')
            self.save(nuclei_mask_np, path, 'nmask.tif')
        except RuntimeError as e:
            print(f"运行错误，不存在该图像++++++++++++++++++++++++ {path}")
            self.output.append(f"运行错误，不存在该图像++++++++++++++++++++++++ {path}")


    def pretreatment(self, image_np, need_CLAHE=False):
        if image_np.mean() * 20 < image_np.max():
            print("该图像的存在较高的亮度，需要进行log对换降低最高亮度值！！！")
            result_image = self.log_image(image_np)
        else:
            result_image = image_np
        result_image = self.gaussian_image(result_image)
        if need_CLAHE:
            # 细胞核图像需要局部增强对比度操作
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            result_image = clahe.apply(result_image)

        # 下采样到 512x512
        result_image = cv2.resize(result_image, (512, 512), interpolation=cv2.INTER_AREA)
        # show_gray_image(result_image)
        return result_image


    def seg_mit_nuclei(self, image_np, factor):
        masks, flows, styles = self.seg_model.eval(image_np,
                                                          diameter=self.seg_diameter / factor,
                                                          flow_threshold=0.4)
        # show_gray_image(masks)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)
        return masks_filtered

    def seg_nuclei(self, image_np, factor):
        masks, flows, styles = self.seg_model.eval(image_np,
                                                   diameter=self.seg_nuclei_diameter / factor,
                                                   flow_threshold=0.4,)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_nuclei_min_diameter / factor,
                                                          max_diameter=self.seg_nuclei_max_diameter / factor)
        # show_gray_image(masks)
        return masks_filtered


    def segmentation(self, image_np):
        """
        使用cellpose进行分割操作
        1. 下采样
        2. 分割
        :return:
        """
        # 分割单细胞区域掩码返回图像
        mit_mask_np = self.seg_mit_nuclei(image_np, self.factor)
        # 分割细胞核掩码返回图像
        nuclei_mask_np = self.seg_nuclei(image_np[:, :, 1], self.factor)

        mit_mask_np, nuclei_mask_np = self.common_mask(mit_mask_np, nuclei_mask_np)
        # show_gray_image(nuclei_mask_np)
        # show_gray_image(mit_mask_np)
        # 还原掩码到原始尺寸
        mit_mask_np = cv2.resize(mit_mask_np.astype(np.uint8), (self.original_width, self.original_height),
                                     interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        nuclei_mask_np = cv2.resize(nuclei_mask_np.astype(np.uint8), (self.original_width, self.original_height),
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
    cell.start(r'D:\data\20250514\EGFR\A549-AFA-2h-d1-c25μm\4')