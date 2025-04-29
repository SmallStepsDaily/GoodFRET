import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ui.tool.grayscale_to_rgb_ui import GrayscaleToRGBUI


class HelpUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.grayscale_to_rgb_ui = None

    def initUI(self):
        # 设置窗口大小
        self.setFixedSize(1980, 1080)

        # 定义QLabel并设置大小和位置
        left_title_label = QLabel("工具栏", self)
        left_title_label.setFont(QFont("Arial", 20, QFont.Bold))
        # 设置QLabel的位置和大小
        left_title_label.setGeometry(20, 20, 300, 50)

        right_title_label = QLabel("功能说明", self)
        right_title_label.setFont(QFont("Arial", 20, QFont.Bold))
        # 设置QLabel的位置和大小
        right_title_label.setGeometry(1000, 20, 300, 50)

        # 整体布局
        main_layout = QHBoxLayout()
        # 设置左右布局之间的间距
        main_layout.setSpacing(50)

        # 左边布局
        left_layout = QVBoxLayout()
        # 设置标签和按钮之间的间距
        left_layout.setSpacing(50)

        # 上方间距
        left_layout.addSpacerItem(QSpacerItem(0, 50, QSizePolicy.Minimum, QSizePolicy.Fixed))

        gray_to_rgb_button = QPushButton("灰度转RGB")
        gray_to_rgb_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                border-radius: 10px;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 25px;
                margin: 10px 2px;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        # 设置按钮大小
        gray_to_rgb_button.setFixedSize(300, 100)
        # 居中添加按钮，使用 Qt 模块的枚举值
        left_layout.addWidget(gray_to_rgb_button, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        gray_to_rgb_button.clicked.connect(self.show_grayscale_to_rgb_ui)

        # 下方间距
        left_layout.addSpacerItem(QSpacerItem(0, 50, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 右边布局
        right_layout = QVBoxLayout()
        # 设置标签和按钮之间的间距
        right_layout.setSpacing(50)

        # 上方间距
        right_layout.addSpacerItem(QSpacerItem(0, 50, QSizePolicy.Minimum, QSizePolicy.Fixed))

        instruction_button = QPushButton("操作说明")
        instruction_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                border-radius: 10px;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 25px;
                margin: 10px 2px;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        # 设置按钮大小
        instruction_button.setFixedSize(300, 100)
        # 居中添加按钮，使用 Qt 模块的枚举值
        right_layout.addWidget(instruction_button, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        instruction_button.clicked.connect(self.show_instruction)

        video_tutorial_button = QPushButton("视频讲解")
        video_tutorial_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                border: none;
                border-radius: 10px;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 25px;
                margin: 10px 2px;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: #e64a19;
            }
        """)
        # 设置按钮大小
        video_tutorial_button.setFixedSize(300, 100)
        # 居中添加按钮，使用 Qt 模块的枚举值
        right_layout.addWidget(video_tutorial_button, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        video_tutorial_button.clicked.connect(self.show_video_tutorial)

        # 下方间距
        right_layout.addSpacerItem(QSpacerItem(0, 50, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 将左右布局添加到整体布局
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 设置主窗口布局
        self.setLayout(main_layout)

    def show_grayscale_to_rgb_ui(self):
        if not self.grayscale_to_rgb_ui:
            self.grayscale_to_rgb_ui = GrayscaleToRGBUI()
        self.grayscale_to_rgb_ui.show()

    def show_instruction(self):
        print("这里可以添加操作说明的逻辑")

    def show_video_tutorial(self):
        print("这里可以添加视频讲解的逻辑")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HelpUI()
    window.show()
    sys.exit(app.exec_())
