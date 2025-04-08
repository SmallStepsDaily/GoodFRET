import sys

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QMetaObject
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QTabWidget, QStatusBar, QPushButton, QHBoxLayout
)
# 在创建 QApplication 之前设置属性
QtCore.QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

class OutputRedirector:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        QMetaObject.invokeMethod(self.text_edit, "append", Qt.QueuedConnection, Q_ARG(str, text))

    def flush(self):
        pass

class ImageProcessingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('GoodFRET')
        self.setGeometry(100, 100, 1280, 800)
        self.center()

        # 设置窗口图标
        self.setWindowIcon(QIcon('logo.jpg'))  # 加载图标文件

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建中部操作区的标签页
        actions = ['数据说明', '名称修改', '图像分割', 'FRET特征提取', '亚细胞器特征提取', '特征分析', '药效分析', '帮助']
        self.tab_widget = QTabWidget()
        for i in range(len(actions)):
            tab = QWidget()
            # 这里可以根据不同的标签页加载不同的界面
            # 假设每个标签页对应的界面在不同的 py 文件中定义
            # 例如，对于 '名称修改' 标签页，假设对应的界面类在 rename_ui.py 文件中定义为 RenameUI
            if actions[i] == '名称修改':
                from ui.rename_ui import RenameUI
                layout = QVBoxLayout()
                rename_ui = RenameUI()
                layout.addWidget(rename_ui)
                tab.setLayout(layout)
            elif actions[i] == '图像分割':
                from ui.segmentation_ui import SegmentationUI
                layout = QVBoxLayout()
                rename_ui = SegmentationUI()
                layout.addWidget(rename_ui)
                tab.setLayout(layout)
            elif actions[i] == '数据说明':
                from ui.description_ui import MarkdownReaderUI
                layout = QVBoxLayout()
                description_ui = MarkdownReaderUI()
                layout.addWidget(description_ui)
                tab.setLayout(layout)
            # 其他标签页类似处理
            self.tab_widget.addTab(tab, actions[i])

        # 布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        central_widget.setLayout(main_layout)

    def center(self):
        # 获取屏幕尺寸
        screen_geometry = QApplication.desktop().screenGeometry()
        # 获取窗口尺寸
        window_geometry = self.geometry()
        # 计算窗口左上角坐标
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        # 移动窗口到计算得到的位置
        self.move(x, y)


def load_window():
    app = QApplication(sys.argv)
    window = ImageProcessingUI()
    window.show()
    sys.exit(app.exec_())
