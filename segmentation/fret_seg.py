import os
import sys
import warnings
import cv2
import numpy as np
import tifffile as tiff
from segmentation.seg import Segmentation, normalize_image, filter_labeled_masks_by_diameter
from cellpose import models

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")

class FRETSegmentation(Segmentation):
    """
    细胞核分割
    """
    def __init__(self,
                 output_redirector=sys.stdout,
                 seg_diameter=200,
                 seg_min_diameter=100,
                 seg_max_diameter=500):
        super().__init__(output_redirector)
        # 读取细胞核图像
        self.channel = [0, 0]
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.Cellpose(gpu=True, model_type='cyto3')
        self.factor = 1
        self.original_width = 2048
        self.original_height = 2048

    def start(self, path):
        # 读取细胞核图像
        dd_image_np = tiff.imread(os.path.join(path, 'DD.tif'))
        # da_image_np = tiff.imread(os.path.join(path, 'DA.tif'))
        aa_image_np = tiff.imread(os.path.join(path, 'AA.tif'))
        # 合并三张图像
        merged_image = np.dstack((dd_image_np, aa_image_np))
        # 转换为灰度图像
        fret_image_np = self.pretreatment(merged_image)
        print("分割细胞核操作 ===================> ", path)
        current_image_np = self.segmentation(fret_image_np)
        # self.show(current_image_np)
        print("保存细胞核操作 ===================> ", path)
        # 保存对应的图像
        self.save(current_image_np, path, 'fret_mask.tif')

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        使用CLAHE方法动态增强512×512图像的对比度

        参数:
        image (np.ndarray): 输入的512×512单通道图像，数据类型应为uint8

        返回:
        np.ndarray: 增强对比度后的512×512图像，数据类型为uint8
        """
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        # 应用CLAHE进行对比度增强
        enhanced_image = clahe.apply(image)
        return enhanced_image

    def pretreatment(self, image):
        self.original_width, self.original_height = image.shape[0], image.shape[1]
        self.factor = image.shape[0] / 512

        merged_image = np.zeros((512, 512), dtype=np.float32)

        for i in range(0, 2):
            # 下采样到 512x512
            image_np = cv2.resize(image[:, :, i], (512, 512), interpolation=cv2.INTER_AREA)
            enhanced_image = self.enhance_contrast(image_np)
            merged_image += enhanced_image.astype(np.float32)

        # 按照 1:1:1 的比例合并图像
        merged_image = merged_image / 2
        result_image = normalize_image(merged_image)
        final_image = cv2.GaussianBlur(result_image, (5, 5), 0)
        return final_image

    def segmentation(self, image_np):
        """
        使用cellpose进行分割操作
        :return:
        """
        masks, flows, styles, diams = self.seg_model.eval(image_np,
                                                          diameter=self.seg_diameter / self.factor,
                                                          channels=self.channel,
                                                          flow_threshold=0.4,
                                                          resample=True,
                                                          do_3D=False)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / self.factor,
                                                          max_diameter=self.seg_max_diameter / self.factor)

        # 还原掩码到原始尺寸
        masks_filtered = cv2.resize(masks_filtered.astype(np.uint8), (self.original_width, self.original_height),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        return masks_filtered


if __name__ == '__main__':
    nuclei = FRETSegmentation()
    nuclei.start(r'D:\data\20250412\BCL2-BAK\MCF7-control-2h-d3-c0μm\7')