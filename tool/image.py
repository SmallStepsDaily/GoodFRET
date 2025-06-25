import cv2
import numpy as np


import cv2
import numpy as np
import ctypes

def show_gray_image(array, window_name='Gray Image', wait_time=0):
    """
    将二维 numpy 数组显示为 OpenCV 灰度图像，并将窗口居中显示

    参数:
        array: 二维 numpy 数组，作为灰度图像数据
        window_name: 显示窗口的名称，默认为 'Gray Image'
        wait_time: 等待按键的时间（毫秒），0 表示无限等待，默认为 0
    """
    # 确保输入数组是二维的
    if array.ndim != 2:
        raise ValueError("输入数组必须是二维的")
    print(array.shape)

    # 转换数据类型为 uint8
    if array.dtype != np.uint8:
        if array.max() > array.min():
            normalized = (array - array.min()) / (array.max() - array.min()) * 255
        else:
            normalized = np.zeros_like(array, dtype=np.float64)
        array_uint8 = normalized.astype(np.uint8)
    else:
        array_uint8 = array.copy()

    # 创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 获取屏幕大小（Windows下使用 ctypes）
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    # 计算窗口位置（图像尺寸）
    img_h, img_w = array_uint8.shape
    pos_x = max((screen_w - img_w) // 2, 0)
    pos_y = max((screen_h - img_h) // 2, 0)

    # 设置窗口位置
    cv2.moveWindow(window_name, pos_x, pos_y)

    # 显示图像
    cv2.imshow(window_name, array_uint8)
    cv2.waitKey(wait_time)
