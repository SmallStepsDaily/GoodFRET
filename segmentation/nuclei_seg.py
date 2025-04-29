import os
import sys
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

class NucleiSegmentation(Segmentation):
    """
    细胞核分割
    """
    def __init__(self,
                 output_redirector=sys.stdout,
                 seg_diameter=120,
                 seg_min_diameter=80,
                 seg_max_diameter=200):
        super().__init__(output_redirector)
        # 读取细胞核图像
        self.channel = [0, 0]
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.Cellpose(gpu=True, model_type='nuclei')

    def start(self, path):
        # 读取细胞核图像
        nuclei_image_np = tiff.imread(os.path.join(path, 'Hoechst.tif'))
        nuclei_image_np = self.pretreatment(nuclei_image_np)
        print("分割细胞核操作 ===================> ", path)
        current_image_np = self.segmentation(nuclei_image_np)
        print("保存细胞核操作 ===================> ", path)
        # 保存对应的图像
        self.save(current_image_np, path, 'n_mask.jpg')

    def pretreatment(self, image):
        result_image = self.log_image(image)
        result_image = self.gaussian_image(result_image)
        return result_image


    def segmentation(self, image_np):
        """
        使用cellpose进行分割操作
        :return:
        """
        # 记录原始图像尺寸
        original_height, original_width = image_np.shape[:2]
        factor = 1
        if original_height == original_width:
            # 下采样到 512x512
            image_np = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_AREA)
            factor = original_height / 512
        masks, flows, styles, diams = self.seg_model.eval(image_np,
                                                          diameter=self.seg_diameter / factor,
                                                          channels=self.channel,
                                                          flow_threshold=0.4,
                                                          resample=True,
                                                          do_3D=False)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)

        if original_height == original_width:
            # 还原掩码到原始尺寸
            masks_filtered = cv2.resize(masks_filtered.astype(np.uint8), (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return masks_filtered


if __name__ == '__main__':
    nuclei = NucleiSegmentation()
    nuclei.start(r'D:\data\qrm\2025.03.26 A549 24H\ALM\6')