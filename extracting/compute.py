import importlib
import os.path
import sys
import numpy as np
import torch
from PIL import Image
from tifffile import tifffile

from ui import Output

"""
FRET 效率计算函数
注意数据加载顺序是 AA、DD、DA
"""
def load_image_to_tensor(image_path, dtype=torch.float):
    """
    加载图像到 GPU 上
    期望输入图像为 二维图像
    """
    img = Image.open(image_path)
    return torch.from_numpy(np.array(img)).type(dtype=dtype)


def load_image_to_numpy(image_path, dtype=np.float32):
    """
    加载图像到 numpy 数组
    期望输入图像为 二维图像

    :param image_path: 图像文件的路径
    :param dtype: 转换后 numpy 数组的数据类型，默认为 np.float32
    :return: 加载并转换后的 numpy 数组
    """
    # 打开图像文件
    img = Image.open(image_path)
    # 将图像转换为 numpy 数组，并指定数据类型
    return np.array(img, dtype=dtype)


class FRETComputer:
    """
    E-FRET 计算
    这套参数测量于2023年9月28日
    """

    def __init__(self,
                 fret_target_name,
                 rc_min=0.0,
                 rc_max=2.5,
                 ed_min=0.0,
                 ed_max=1.0,
                 a: float = 0.150332,
                 b: float = 0.001107,
                 c: float = 0.000561,
                 d: float = 0.780016,
                 G: float = 5.494216,
                 k: float = 0.432334,
                 expose_times: tuple = (300, 300, 300),
                 output_redirector=Output(),
                 need_Ed=True,
                 need_Rc=True,
                 need_Fp=True,
                 need_Rc_Ed=False,
                 ):
        """
        :param a: 校正因子a
        :param b: 校正因子b
        :param c: 校正因子c
        :param d: 校正因子d
        :param G: 校正因子G
        :param k: 校正因子k
        :param expose_times: 三通道曝光时间
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.G = G
        self.k = k
        self.expose_times = expose_times
        self.current_sub_path = ''                  # 当前处理的子文件夹
        self.fret_target_name = fret_target_name    # FRET 靶点名称
        self.rc_min = rc_min
        self.rc_max = rc_max
        self.ed_min = ed_min
        self.ed_max = ed_max

        # 运行时参数
        self.image_AA = None
        self.image_DD = None
        self.image_DA = None
        self.mask = None
        self.image_Ed = None
        self.image_Rc = None
        self.fret_mask = None
        self.nuclei_mask = None

        # 统计三通道的背景阈值
        self.background_noise_values = {}

        # 判断是否需要根据亚细胞器进行FRET特征提取
        self.extract_organelle = True

        # 特征类型选择
        self.need_Ed = need_Ed
        self.need_Rc = need_Rc
        self.need_Fp = need_Fp
        self.need_Rc_Ed = need_Rc_Ed

        # 命令行输出到文本框内
        self.output = output_redirector

    def start(self, sub_path):
        """
        利用 pytorch 进行 fret 效率计算操作
        """
        print(f"FRET计算处理 ============================================> {sub_path}")
        self.output.append(f"FRET计算处理 ============================================> {sub_path}")
        self.current_sub_path = sub_path
        # 加载图像为 tensor 并移动到相同设备（假设都在 CPU 或都在 GPU）
        image_AA = load_image_to_tensor(os.path.join(sub_path, 'AA.tif'))
        image_DD = load_image_to_tensor(os.path.join(sub_path, 'DD.tif'))
        image_DA = load_image_to_tensor(os.path.join(sub_path, 'DA.tif'))

        mask = None
        # 在这里需要判断，掩码是基于亚细胞器染料通道的分割掩码还是基于FRET三通道的掩码
        if os.path.exists(os.path.join(sub_path, 'mmask.tif')):
            # 基于亚细胞器染料通道的分割掩码
            mask = load_image_to_tensor(os.path.join(sub_path, 'mmask.tif'), dtype=torch.uint8)
            self.extract_organelle = True
        elif os.path.exists(os.path.join(sub_path, 'fret_mask.tif')):
            # 基于FRET三通道的掩码
            mask = load_image_to_tensor(os.path.join(sub_path, 'fret_mask.tif'), dtype=torch.uint8)
            self.extract_organelle = False

        # 记录数据
        self.image_DD = image_DD
        self.image_AA = image_AA
        self.image_DA = image_DA
        self.mask = mask

        # 计算背景噪声 并且FRET三通道减去对应的背景噪声
        image_AA, image_AA_template, self.background_noise_values['AA'] = self.subtract_background_noise(image_AA, current_expose_times=self.expose_times[0])
        image_DD, image_DD_template, self.background_noise_values['DD'] = self.subtract_background_noise(image_DD, current_expose_times=self.expose_times[1])
        image_DA, image_DA_template, self.background_noise_values['DA'] = self.subtract_background_noise(image_DA, current_expose_times=self.expose_times[2])

        # 添加三通道有效模板 三通道值全部必须为正
        effective_template = image_AA_template * image_DD_template * image_DA_template

        # 计算 Fc 图像
        Fc = image_DA - self.a * (image_AA - self.c * image_DD) - self.d * (image_DD - self.b * image_AA)
        Fc[Fc < 0] = 0

        # 计算 Ed 效率以及 Rc 浓度值
        Ed = Fc / (Fc + self.G * image_DD + 1e-7) * effective_template
        Rc = ((self.k * image_AA) / (image_DD + Fc / self.G + 1e-12)) * effective_template

        # TODO 保存Ed效率图 保存为 TIFF 文件（可以选择其他格式，如 PNG），并设置保存参数以保留浮点数精度
        tifffile.imwrite(os.path.join(self.current_sub_path, 'Ed.tif'), Ed.numpy())
        tifffile.imwrite(os.path.join(self.current_sub_path, 'Rc.tif'), Rc.numpy())

        # 筛选合理的单细胞掩码图像
        image_Ed, fmask = self.filter_cell_region(Ed, mask)
        image_Rc = Rc * torch.where(fmask > 1, torch.tensor(1, device=fmask.device), fmask)

        # 记录数据
        self.image_Ed = image_Ed
        self.image_Rc = image_Rc

        self.fret_mask = fmask.type(dtype=torch.uint8)
        # 保存为图像文件 TODO
        tifffile.imwrite(os.path.join(self.current_sub_path, 'fmask.tif'), self.fret_mask.numpy().astype(np.uint8))
        # TODO 开始提取效率特征
        start_extraction = importlib.import_module(f'extracting.{self.fret_target_name}')
        result = start_extraction.start(self)
        print(f"FRET计算完成 ============================================> {sub_path}")
        self.output.append(f"FRET计算完成 ============================================> {sub_path}")
        return result

    @staticmethod
    def filter_cell_region(image_ED, mask):
        unique_labels = torch.unique(mask)
        # 排除背景标签 0
        unique_labels = unique_labels[unique_labels != 0]
        filtered_mask = torch.zeros_like(mask)
        # 重新编码过滤后的掩码，使有效区域从1开始
        new_label = 1
        for label in unique_labels:
            label_mask = mask == label

            region_pixels = image_ED[label_mask]
            non_zero_count = torch.sum(region_pixels > 0)
            total_pixels = region_pixels.numel()
            # 需要满足有效效率区域大于百分之50的情况，否则容易存在噪声
            if non_zero_count <= total_pixels * 0.5:
                filtered_mask[label_mask] = 0
            else:
                filtered_mask[label_mask] = new_label
                new_label += 1

        # 直接在乘法运算中进行条件判断和赋值
        final_ed_result = image_ED * torch.where(filtered_mask > 1, torch.tensor(1, device=filtered_mask.device),
                                              filtered_mask)

        return final_ed_result, filtered_mask


    @staticmethod
    def filter_overexpose(image_AA, image_DD):
        """
        被弃用，放在后面流程上进行处理
        筛选过曝的细胞
        1. AA荧光强度大于DD通道十倍设置为0
        """
        # 创建一个与AA和DD相同形状的全1张量
        result = torch.ones_like(image_AA)
        # 找到满足条件的点（AA比DD的强度大10倍）
        condition = image_AA > 10 * image_DD
        # 将满足条件的点设置为0
        result[condition] = 0

        return result

    @staticmethod
    def subtract_background_noise(image, background_threshold=1.2, current_expose_times=300):
        background_flat = image.squeeze().flatten().numpy()

        # 统计直方图
        hist, bin_edges = np.histogram(background_flat, bins=np.arange(1, 2001, 1))

        # 找到频率最高像素值对应的索引
        most_frequent_index = np.argmax(hist)

        # 设定低峰索引上限
        max_low_peak_index = int(most_frequent_index * background_threshold)

        # 从最高峰之后开始寻找低峰
        low_peak_index = most_frequent_index + 1
        while low_peak_index < len(hist) - 1 and low_peak_index <= max_low_peak_index:
            if hist[low_peak_index] < hist[low_peak_index - 1] and hist[low_peak_index] < hist[low_peak_index + 1]:
                break
            low_peak_index += 1

        # 如果没找到合适低峰，使用 most_frequent_index 的 1.2 倍对应的值
        if low_peak_index > max_low_peak_index:
            low_peak_index = max_low_peak_index

        # 获取低峰对应的像素值作为背景噪声
        background_noise = bin_edges[low_peak_index] * background_threshold

        # 将图像减去背景噪声并进行掩码屏蔽
        noise_removed_tensor = image - bin_edges[low_peak_index]
        noise_removed_tensor[noise_removed_tensor < 0] = 0
        noise_removed_tensor[noise_removed_tensor > 50000] = 0

        # 计算该图像的有效区域的模板
        template_tensor = image - background_noise
        template_tensor[template_tensor > 0] = 1
        template_tensor[template_tensor <= 0] = 0

        # 除以曝光时间
        noise_removed_tensor = noise_removed_tensor / current_expose_times

        # 返回降噪结果
        return noise_removed_tensor, template_tensor, background_noise / current_expose_times


if __name__ == "__main__":
    # EGFR 靶点验证
    fret = FRETComputer('egfr_grb2', expose_times=(300, 300, 300))
    fret.start(r'D:\data\hql\2025.05.26 fret hoechst mito BF\H1975-Osi-2h-d4-c4μm\9')
    # BAX 靶点验证
    # fret = FRETComputer('bax_bak', expose_times=(300, 300, 300))
    # # D:\data\20250412\BCLXL-BAK\MCF7-A133-2h-d1-c60μm\6
    # # D:\data\20250412\BCLXL-BAK\MCF7-control-2h-d3-c0μm\4
    # # D:\data\20250513\BCLXL-BAK\MCF7-control-2h-d3-c0μm\10
    # fret.start(r'D:\data\20250515\BCLXL\MCF7-A199-6h-d1-c90μm\7')