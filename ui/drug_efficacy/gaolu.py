from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget


class GaoLuWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('高璐药效函数窗口')
        layout = QVBoxLayout()
        label = QLabel('这是高璐药效函数窗口')
        layout.addWidget(label)
        self.setLayout(layout)
