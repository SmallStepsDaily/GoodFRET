from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import QTextEdit

class Output:
    def append(self, text):
        pass


class TextUpdateHandler(QObject):
    """线程安全的文本更新处理器"""
    update_signal = pyqtSignal(str)  # 定义信号

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self.text_edit = text_edit
        self.update_signal.connect(self.append_text, Qt.QueuedConnection)

    def append(self, text: str):
        """从任意线程安全调用的方法"""
        self.update_signal.emit(text)

    def append_text(self, text: str):
        """在主线程中执行的槽函数"""
        self.text_edit.append(text)