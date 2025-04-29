
from ui.main_ui import load_window
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 文件夹的路径
data_folder = os.path.join(current_dir, 'data/model')
# 设置 CELLPOSE_LOCAL_MODELS_PATH 环境变量
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = data_folder

if __name__ == '__main__':
    # 启动窗口
    load_window()