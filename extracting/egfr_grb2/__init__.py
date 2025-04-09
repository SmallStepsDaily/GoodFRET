import os

from extracting.compute import load_image_to_tensor


def start(fret):
    """
    fret 具有如下参数可以调用
        self.current_sub_path = ''                  # 当前处理的子文件夹
        self.fret_target_name = fret_target_name    # FRET 靶点名称
        self.rc_min = rc_min
        self.rc_max = rc_max
        self.ed_min = ed_min
        self.ed_max = ed_max
        self.image_AA = None
        self.image_DD = None
        self.image_DA = None
        self.mask = None
        self.image_Ed = None
        self.image_Rc = None
        self.fret_mask = None
        self.nuclei_mask = None

    提取特征：
    1. 效率特征
    2. 浓度特征
    3. 共定位特征

    需要判断是否存在细胞核图像，决定是否掩码细胞核提取特征
    """
    print(fret.current_sub_path)
    nuclei_mask = None
    if os.path.exists(os.path.join(fret.current_sub_path, 'nmask.jpg')):
        nuclei_mask = load_image_to_tensor(os.path.join(fret.current_sub_path, 'nmask.jpg'))

