import numpy as np
import pandas as pd


def calculate_fluorescence_ratio(image_np, outer_mask, inner_mask):
    """
    计算 outer_mask / inner_mask 荧光比例
    """
    # 获取唯一的ObjectNumber
    object_numbers = np.unique(outer_mask)
    object_numbers = object_numbers[object_numbers > 0]  # 排除背景（掩码值为0）

    data = []
    for object_number in object_numbers:
        # 获取当前ObjectNumber对应的outer_mask和inner_mask
        current_outer_mask = (outer_mask == object_number)
        current_inner_mask = (inner_mask == object_number)

        # 计算outer_inner_mask
        outer_inner_mask = current_outer_mask & ~current_inner_mask

        # 统计outer_inner_mask对应的image_np平均荧光值
        if np.any(outer_inner_mask):
            outer_intensity = np.mean(image_np[outer_inner_mask])
        else:
            outer_intensity = np.nan

        # 统计inner_mask内的image_np平均荧光值
        if np.any(current_inner_mask):
            inner_intensity = np.mean(image_np[current_inner_mask])
        else:
            inner_intensity = np.nan

        # 计算荧光比例
        if inner_intensity == 0 or np.isnan(inner_intensity):
            intensity_ratio = np.nan
        else:
            intensity_ratio = outer_intensity / inner_intensity

        # 记录数据
        data.append([object_number, outer_intensity, inner_intensity, intensity_ratio])

    # 创建pandas DataFrame
    columns = ['ObjectNumber', 'outer_intensity_mean', 'inner_intensity_mean', 'intensity_ratio']
    df = pd.DataFrame(data, columns=columns)

    return df
