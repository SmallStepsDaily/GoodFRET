"""
FRET分析
"""
import base64


class FRETCharacterizationValue:

    def __init__(self, data_df):
        self.data = data_df
        self.metadata_columns = [col for col in self.data.columns if col.startswith('Metadata_') or col == 'ObjectNumber']

    def start(self, *args, **kwargs):
        pass

    def compute(self, *args, **kwargs):
        pass

    def draw_plt(self, *args, **kwargs):
        pass

def save_base64_with_prefix(base64_str, output_path):
    """
    保存带前缀的Base64图像（如"data:image/png;base64,..."）
    :param base64_str: 包含前缀的完整Base64字符串
    :param output_path: 保存路径（如"image.png"）
    """
    try:
        # 检查是否包含前缀
        if not base64_str.startswith('data:image/png;base64,'):
            raise ValueError("无效的Base64字符串，缺少前缀")

        # 提取纯Base64数据（去除前缀）
        pure_base64 = base64_str.split('base64,')[1]

        # 补全填充字符（解决Incorrect padding错误）
        missing_padding = len(pure_base64) % 4
        if missing_padding != 0:
            pure_base64 += '=' * (4 - missing_padding)

        # 解码并保存为PNG文件
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(pure_base64))

        print(f"图像已成功保存至：{output_path}")

    except Exception as e:
        print(f"保存失败：{str(e)}")
