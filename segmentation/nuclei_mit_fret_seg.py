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

class MitNucleiFRETSegmentation(Segmentation):
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
                 DD_scaler=1,
                 AA_scaler=1,
                 weight=0.5,
                 output_redirector=Output()):
        super().__init__(output_redirector)
        # FRET 通道融合比例
        self.weight = weight

        # 针对DD、AA通道的阈值判断
        self.DD_scaler = DD_scaler
        self.AA_scaler = AA_scaler

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
        self.seg_nuclei_model = models.CellposeModel(gpu=True)

    def start(self, path):
        try:
            # 读取细胞核和线粒体图像
            mit_image_np = tiff.imread(os.path.join(path, 'Mit.tif'))
            nuclei_image_np = tiff.imread(os.path.join(path, 'Hoechst.tif'))
            # 读取细胞核图像
            dd_image_np = normalize_image(tiff.imread(os.path.join(path, 'DD.tif')))
            aa_image_np = normalize_image(tiff.imread(os.path.join(path, 'AA.tif')))
            fret_image_np = cv2.addWeighted(aa_image_np, self.weight, dd_image_np, 1 - self.weight, 0)
            fret_image_np = cv2.resize(fret_image_np, (512, 512), interpolation=cv2.INTER_AREA)
            # 记录原始图像尺寸
            self.original_height, self.original_width = mit_image_np.shape
            self.factor = self.original_height / 512
            # 将两个通道进行组合操作
            mit_image_np = self.pretreatment(mit_image_np)
            nuclei_image_np = self.pretreatment(nuclei_image_np, need_CLAHE=True)
            current_image_np = np.stack([nuclei_image_np, mit_image_np, fret_image_np], axis=0)

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
            result_image = normalize_image(image_np)
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
        masks, flows, styles = self.seg_model.eval(image_np)
        # show_gray_image(masks)
        masks_filtered = filter_labeled_masks_by_diameter(masks,
                                                          min_diameter=self.seg_min_diameter / factor,
                                                          max_diameter=self.seg_max_diameter / factor)
        return masks_filtered

    def seg_nuclei(self, image_np, factor):
        masks, flows, styles = self.seg_nuclei_model.eval(image_np,
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
        # 分割线粒体区域
        mit_nuclei_np = np.stack((image_np[1], image_np[0]), axis=0)
        mit_mask_np = self.seg_mit_nuclei(mit_nuclei_np, self.factor)
        # TODO 还需要进一步进行修改
        # 基于线粒体区域，拓展得出FRET单细胞区域
        fret_mit_np = np.maximum(image_np[1], image_np[2])
        fret_mit_nuclei_np = np.stack((fret_mit_np, image_np[0]), axis=0)
        fret_mask_np = self.seg_mit_nuclei(fret_mit_nuclei_np, self.factor)

        # 分割细胞核掩码返回图像
        nuclei_mask_np = self.seg_nuclei(image_np[0], self.factor)
        mit_mask_np, nuclei_mask_np = self.common_mask(mit_mask_np, nuclei_mask_np)

        # 还原掩码到原始尺寸
        mit_mask_np = cv2.resize(mit_mask_np.astype(np.uint8), (self.original_width, self.original_height),
                                     interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        nuclei_mask_np = cv2.resize(nuclei_mask_np.astype(np.uint8), (self.original_width, self.original_height),
                                     interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return mit_mask_np, nuclei_mask_np

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
    """
    测试验证代码
    """
    import torch
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ON!")
    else:
        print("CUDA is NOT available. GPU is OFF.")

    cell = MitNucleiFRETSegmentation()
    cell.start(r'D:\data\20250514\EGFR\A549-AFA-2h-d1-c25μm\4')