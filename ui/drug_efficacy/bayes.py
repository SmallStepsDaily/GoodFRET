from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel


class PengChuanBayesWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('彭川贝叶斯模型窗口')
        layout = QVBoxLayout()
        label = QLabel('这是彭川贝叶斯模型窗口')
        layout.addWidget(label)
        self.setLayout(layout)