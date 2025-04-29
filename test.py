import cv2
import numpy as np


def custom_color_map(gray_image):
    height, width = gray_image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gray_value = gray_image[y, x]
            r = int(gray_value * (0 / 255))
            g = int(gray_value * (250 / 255))
            b = int(gray_value * (16 / 255))
            colored_image[y, x] = [b, g, r]
    return colored_image


def log_image(image):
    """
    对输入的图像进行预处理，使用对数函数降维
    """
    # 转换为浮点数类型
    image_float = image.astype(np.float32)
    # 对数变换
    image_log = np.log1p(image_float)
    # 归一化到0-255
    image_normalized = cv2.normalize(image_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return image_normalized


def combine_images(image_path1, image_path2):
    try:
        # 读取两张TIFF灰度图片
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None:
            print("无法读取图像，请检查图像路径。")
            return

        # 确保两张图像尺寸相同
        if image1.shape != image2.shape:
            print("两张图像尺寸不一致，请使用相同尺寸的图像。")
            return

        # 对两张图像进行预处理
        processed_image1 = log_image(image1)
        processed_image2 = log_image(image2)

        processed_image1 = remove_background(processed_image1)
        processed_image2 = remove_background(processed_image2)
        # 创建一个全零的绿色通道
        green_channel = np.zeros_like(processed_image1)

        # 组合三个通道成一张彩色图片
        combined_image = cv2.merge([processed_image1, green_channel, processed_image2])

        # 缩放到512x512大小
        img_resized = cv2.resize(combined_image, (512, 512), interpolation=cv2.INTER_AREA)
        # 显示组合后的图像
        cv2.imshow('Combined Image', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存组合后的图像
        cv2.imwrite('combined_image.tif', combined_image)
        print("组合后的图像已保存为 combined_image.tif")

    except Exception as e:
        print(f"发生错误: {e}")

def remove_background(img_normalized):
    # 背景去除（使用直方图统计）
    hist = cv2.calcHist([img_normalized], [0], None, [256], [0, 256])
    threshold = np.argmax(hist) * 1.3  # 找到直方图中最大值对应的灰度级
    _, img_thresholded = cv2.threshold(img_normalized, threshold, 255, cv2.THRESH_BINARY)

    # 减去整体阈值
    img_equalized = np.clip(img_normalized - threshold, 0, 255).astype(np.uint8)
    return img_equalized

def process_image(image_path):
    try:
        # 读取TIFF图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("无法读取图像，请检查图像路径。")
            return

        # 确保图像是2048x2048
        if img.shape != (2048, 2048):
            print("图像尺寸不是2048x2048。")
            return

        # 对数变换
        img_log = np.log1p(img.astype(np.float32))

        # 归一化到0-255
        img_normalized = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        img_equalized = remove_background(img_normalized)

        # 使用自定义颜色映射
        img_colored = custom_color_map(img_equalized)

        # 保存组合后的图像
        cv2.imwrite('DD_image.tif', img_colored)

        # 缩放到512x512大小
        img_resized = cv2.resize(img_colored, (512, 512), interpolation=cv2.INTER_AREA)

        # 显示图像
        cv2.imshow('Processed Image', img_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"发生错误: {e}")

def combine_images_opencv(img_path1, img_path2):
    # 读取两张TIFF图像
    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    # 检查图像是否成功读取
    if img1 is None or img2 is None:
        print("无法读取图像，请检查图像路径。")
        return

    # 确保两张图像尺寸相同
    if img1.shape != img2.shape:
        print("两张图像尺寸不一致，请使用相同尺寸的图像。")
        return

    # 组合图像（这里以简单的平均组合为例）
    combined_img = (img1.astype(np.float32) + img2.astype(np.float32)) / 2
    combined_img = combined_img.astype(np.uint8)

    # 显示组合后的图像
    cv2.imshow('Combined Image', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存组合后的图像
    cv2.imwrite('combined_image_opencv.tif', combined_img)
    print("组合后的图像已保存为 combined_image_opencv.tif")

if __name__ == "__main__":
    # image_path = r"C:\Users\pengs\Downloads\1\DD.tif"  # 请替换为你的TIFF图像路径
    # process_image(image_path)
    # combine_images(r"C:\Users\pengs\Downloads\1\Hoechst.tif", r"C:\Users\pengs\Downloads\1\Mit.tif")
    combine_images_opencv(r"C:\Users\pengs\Downloads\1\combined_image-1.tif", r"C:\Users\pengs\Downloads\1\DD_image-1.tif")