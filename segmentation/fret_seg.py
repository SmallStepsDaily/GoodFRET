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
        da_image_np = tiff.imread(os.path.join(path, 'DA.tif'))
        aa_image_np = tiff.imread(os.path.join(path, 'AA.tif'))

        # 合并三张图像
        merged_image = np.dstack((dd_image_np, da_image_np, aa_image_np))
        # 转换为灰度图像
        fret_image_np = self.pretreatment(merged_image)

        print("分割细胞核操作 ===================> ", path)
        current_image_np = self.segmentation(fret_image_np)
        print("保存细胞核操作 ===================> ", path)
        # 保存对应的图像
        self.save(current_image_np, path, 'fmask.jpg')

    def pretreatment(self, image):
        self.original_width, self.original_height = image.shape[0], image.shape[1]
        self.factor = image.shape[0] / 512

        result_image = self.log_image(image)

        # 自适应去噪（使用自适应阈值）
        merged_image = np.zeros((512, 512), dtype=np.float32)

        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(16, 16))

        for i in range(0, 3):
            # 下采样到 512x512
            image_np = cv2.resize(result_image[:, :, i], (512, 512), interpolation=cv2.INTER_AREA)
            # 应用CLAHE
            thread_image = clahe.apply(image_np)
            merged_image += thread_image.astype(np.float32)

        # 按照 1:1:1 的比例合并图像
        merged_image = merged_image / 3

        # 将结果转换回 uint8 类型
        merged_image = merged_image.astype(np.uint8)
        final_image = cv2.GaussianBlur(merged_image, (5, 5), 0)
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
    nuclei.start(r'D:\data\20250412\BCL2-BAK\A133-6H-IC50\5')