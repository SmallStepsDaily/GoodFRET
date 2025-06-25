import cv2
import numpy as np
from cellpose import models
from cellpose.io import imread

from segmentation.seg import normalize_image
from tool.image import show_gray_image

"""
用于测试的文件
"""


model = models.CellposeModel(gpu=True)
path = r'D:\data\20250515\BCLXL\MCF7-control-6h-d3-c0μm\7'
file_1 = f"{path}\AA.tif"
image_1 = normalize_image(imread(file_1))
image_np_1 = cv2.resize(image_1, (512, 512), interpolation=cv2.INTER_AREA)

# file_2 = f"{path}\DA.tif"
# image_2 = normalize_image(imread(file_2))
# image_np_2 = cv2.resize(image_2, (512, 512), interpolation=cv2.INTER_AREA)


file_3 = f"{path}\DD.tif"
image_3 = normalize_image(imread(file_3))
image_np_3 = cv2.resize(image_3, (512, 512), interpolation=cv2.INTER_AREA)

# image_np = np.stack((image_np_1, image_np_3), axis=0)
image_np = image_np_1 * 0.5 + image_np_3 * 0.5
# image_np = np.stack((image_np, image_np_1, image_np_3), axis=0)
show_gray_image(image_np)

# image_np = cv2.GaussianBlur(image_np, (5, 5), 2)

masks, flows, styles = model.eval(image_np, flow_threshold=0.4, cellprob_threshold=0)
print(masks.shape)
show_gray_image(masks)