import cv2
import numpy as np


def show_gray_image(array, window_name='Gray Image', wait_time=0):
    """
    将二维 numpy 数组显示为 OpenCV 灰度图像

    参数:
        array: 二维 numpy 数组，作为灰度图像数据
        window_name: 显示窗口的名称，默认为 'Gray Image'
        wait_time: 等待按键的时间（毫秒），0 表示无限等待，默认为 0

    返回:
        None
    """
    # 确保输入数组是二维的
    if array.ndim != 2:
        raise ValueError("输入数组必须是二维的")
    print(array.shape)
    # 转换数据类型为 uint8（OpenCV 灰度图像格式）
    if array.dtype != np.uint8:
        # 归一化到 0-255 范围
        if array.max() > array.min():
            normalized = (array - array.min()) / (array.max() - array.min()) * 255
        else:
            normalized = np.zeros_like(array, dtype=np.float64)
        array_uint8 = normalized.astype(np.uint8)
    else:
        array_uint8 = array.copy()

    # 显示图像
    cv2.imshow(window_name, array_uint8)

    # 等待按键事件
    cv2.waitKey(wait_time)