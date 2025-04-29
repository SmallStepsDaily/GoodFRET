import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QLabel
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve

from ui.drug_efficacy.bayes import PengChuanBayesWindow
from ui.drug_efficacy.gaolu import GaoLuWindow


class FadeInWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()


class DrugEfficacyAnalysisUI(FadeInWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口图标
        self.setWindowIcon(QIcon('path_to_your_icon.ico'))

        self.setWindowTitle('药物疗效分析界面')
        # 设置窗口背景颜色
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('#f0f4f8'))
        self.setPalette(palette)

        layout = QVBoxLayout()

        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #45a049;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            QPushButton:pressed {
                background-color: #3e8e41;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        """

        btn_gaolu = QPushButton('gl药效函数')
        btn_gaolu.setStyleSheet(button_style)
        btn_gaolu.clicked.connect(self.open_gaolu_window)
        layout.addWidget(btn_gaolu)

        btn_pengchuan = QPushButton('pc贝叶斯模型')
        btn_pengchuan.setStyleSheet(button_style)
        btn_pengchuan.clicked.connect(self.open_pengchuan_window)
        layout.addWidget(btn_pengchuan)

        # 设置布局边距和间距
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 添加标题标签
        title_label = QLabel('药物疗效分析系统')
        title_font = QFont('Segoe UI', 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333;")
        layout.insertWidget(0, title_label)

        self.setLayout(layout)

    def open_gaolu_window(self):
        self.gaolu_window = GaoLuWindow()
        self.gaolu_window.show()

    def open_pengchuan_window(self):
        self.pengchuan_window = PengChuanBayesWindow()
        self.pengchuan_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrugEfficacyAnalysisUI()
    window.show()
    sys.exit(app.exec_())
