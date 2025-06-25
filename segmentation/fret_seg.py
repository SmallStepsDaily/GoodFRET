import os
import warnings
import cv2
import numpy as np
from cellpose.io import imread
from segmentation.seg import Segmentation, normalize_image, filter_labeled_masks_by_diameter
from cellpose import models

from tool.image import show_gray_image
from ui import Output

# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")


class FRETSegmentation(Segmentation):
    """
    细胞核分割
    """

    def __init__(self,
                 output_redirector=Output(),
                 seg_diameter=200,
                 seg_min_diameter=80,
                 seg_max_diameter=500,
                 DD_scaler=2,
                 AA_scaler=2,
                 weight=0.5):
        super().__init__(output_redirector)

        # 针对DD、AA通道的阈值判断
        self.DD_scaler = DD_scaler
        self.AA_scaler = AA_scaler

        # 分割单细胞阈值大小判断
        self.seg_diameter = seg_diameter
        self.seg_min_diameter = seg_min_diameter
        self.seg_max_diameter = seg_max_diameter
        self.seg_model = models.CellposeModel(gpu=True)
        self.factor = 1
        self.original_width = 2048
        self.original_height = 2048
        self.weight = weight

    def start(self, path):
        # 读取细胞核图像
        dd_image_np = normalize_image(imread(os.path.join(path, 'DD.tif')))
        aa_image_np = normalize_image(imread(os.path.join(path, 'AA.tif')))
        # 合并图像
        merged_image = np.stack((aa_image_np, dd_image_np), axis=0)
        # 转换为灰度图像
        fret_image_np = self.pretreatment(merged_image)
        print(f"分割FRET细胞操作 ===================> {path}")
        self.output.append(f"分割FRET细胞操作 ===================> {path}")
        current_image_np = self.segmentation(fret_image_np)
        # 获取到的图像还需要在进行过滤，判断对应的荧光条件是否符合
        mask = self.filter_mask_by_intensity(current_image_np, merged_image)
        # show_gray_image(mask)
        print(f"保存细胞操作 ===================> {path}")
        self.output.append(f"保存FRET细胞操作 ===================> {path}")
        # 保存对应的图像
        self.save(mask, path, 'fret_mask.tif')

    def pretreatment(self, image):
        self.original_width, self.original_height = image.shape[1], image.shape[2]
        self.factor = image.shape[1] / 512

        merged_image = np.zeros((2, 512, 512), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        for i in range(0, 2):
            # 下采样到 512x512
            image_np = cv2.resize(image[i, :, :], (512, 512), interpolation=cv2.INTER_AREA)
            clahe_image_np = clahe.apply(image_np)
            blur_image_np = cv2.GaussianBlur(image_np, (5, 5), sigmaX=1)
            merged_image[i, :, :] = cv2.addWeighted(clahe_image_np, 0.8, blur_image_np, 0.2, 0)  # 融合增强图与模糊图

        merged_image = merged_image[0, :, :] * self.weight + merged_image[1, :, :] * (1 - self.weight)
        show_gray_image(merged_image)
        return merged_image

    def segmentation(self, image_np):
        """
        使用cellpose进行分割操作
        :return:
        """
        masks, flows, styles = self.seg_model.eval(image_np,
                                                   flow_threshold=0.4,
                                                   cellprob_threshold=0)
        # show_gray_image(masks)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / self.factor,
                                                          max_diameter=self.seg_max_diameter / self.factor)
        # 还原掩码到原始尺寸
        masks_filtered = cv2.resize(masks_filtered, (self.original_width, self.original_height), interpolation=cv2.INTER_NEAREST)
        return masks_filtered


    def filter_mask_by_intensity(self, mask, merged_image, channels=None):
        """
        根据单细胞区域在merged_image中的亮度过滤mask图像

        参数:
        mask: 输入的mask图像，背景为0，单细胞区域为大于1的整数
        merged_image: 合并后的图像，用于计算亮度
        channels: 需要考虑的通道索引列表，默认为[0, 1]，即只考虑第0和第1通道

        返回:
        filtered_mask: 过滤后的mask图像，保留满足条件的细胞并重新编码
        """
        # 确保输入是numpy数组
        if channels is None:
            channels = [0, 1]
        mask = np.array(mask)
        merged_image = np.array(merged_image)

        # 获取所有细胞的标签
        cell_labels = np.unique(mask)
        cell_labels = cell_labels[cell_labels > 0]  # 排除背景(0)

        # 计算每个通道的背景平均亮度
        background_mask = (mask == 0)
        channel_background_intensities = {}

        for channel in channels:
            channel_pixels = merged_image[channel][background_mask]
            channel_background_intensities[channel] = np.mean(channel_pixels)

        # 用于存储满足条件的细胞
        valid_cells = []

        # 检查每个细胞在每个通道的亮度
        for label in cell_labels:
            cell_mask = (mask == label)
            cell_valid = True

            # 检查细胞在每个通道的亮度是否满足条件
            for channel in channels:
                channel_pixels = merged_image[channel][cell_mask]
                channel_cell_intensity = np.mean(channel_pixels)
                # 针对AA通道和DD通道的性质，设置不同的阈值判断
                if channel == 0:
                    scaler = self.AA_scaler
                else:
                    scaler = self.DD_scaler
                # 如果细胞在任何通道不满足条件，则标记为无效
                if channel_cell_intensity <= scaler * channel_background_intensities[channel]:
                    cell_valid = False
                    break

            # 如果细胞在所有通道都满足条件，则保留
            if cell_valid:
                valid_cells.append(label)

        # 创建新的mask，只包含满足条件的细胞
        filtered_mask = np.zeros_like(mask, dtype=np.uint8)

        # 重新编码细胞标签
        for i, label in enumerate(valid_cells, 1):
            filtered_mask[mask == label] = i
        return filtered_mask


if __name__ == '__main__':
    nuclei = FRETSegmentation()
    nuclei.start(r'D:\data\20250513\BCLXL-BAK\MCF7-A133-2h-d2-c40μm\8')
