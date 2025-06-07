import os
import shutil
import sys
import subprocess
import webbrowser

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from main import current_dir


class FileOpenThread(QThread):
    finished = pyqtSignal()
    error_occurred = pyqtSignal()

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            process = {}
            if os.name == 'nt':  # Windows 系统
                process = subprocess.Popen(['start', '', self.file_path], shell=True)
            elif os.name == 'posix':  # Linux 或 macOS 系统
                process = subprocess.Popen(['open', self.file_path])
            process.wait()
        except Exception as e:
            print(f"发生错误: {e}")
            self.error_occurred.emit()
        finally:
            self.finished.emit()


class PhenotypeExtractionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.buttons = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 定义QLabel并设置大小和位置
        title_label = QLabel("CellProfiler 管道", self)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        # 设置QLabel的位置和大小
        title_label.setGeometry(20, 20, 1000, 50)

        button_texts = ["线粒体明场", "FRET明场", "线粒体细胞核明场", "细胞核明场", "线粒体细胞核明场FRET"]
        button_colors = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#F0D93D"]
        button_functions = [self.mitochondria_brightfield, self.fret_brightfield,
                            self.mitochondria_nucleus_brightfield, self.nucleus_brightfield,
                            self.mitochondria_nucleus_brightfield_FRET]

        for text, color, func in zip(button_texts, button_colors, button_functions):
            button = QPushButton(text)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: none;
                    border-radius: 10px;
                    color: white;
                    font-size: 25px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    padding: 15px 30px;
                    margin: 5px;
                    transition: background-color 0.3s;
                }}
                QPushButton:hover {{
                    background-color: {self.lighten_color(color, 0.2)};
                }}
            """)
            layout.addWidget(button)
            button.clicked.connect(func)
            self.buttons.append(button)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #F8F9FA;")

    def execute_file_operation(self, source_file_name, operation_name):
        self.disable_buttons()
        print(f"执行{operation_name}相关运算逻辑")
        try:
            source_file = os.path.join(current_dir, f'data/pipline/{source_file_name}')
            target_folder = os.path.join(current_dir, 'data/cache')
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            shutil.copy2(source_file, target_folder)
            target_file = os.path.join(target_folder, source_file_name)
            self.thread = FileOpenThread(target_file)
            self.thread.finished.connect(self.enable_buttons)
            self.thread.error_occurred.connect(self.show_error_message)
            self.thread.start()
        except Exception as e:
            print(f"发生错误: {e}")
            self.enable_buttons()
            self.show_error_message()

    def mitochondria_brightfield(self):
        self.execute_file_operation('Mit_BF.cpproj', "线粒体明场")

    def fret_brightfield(self):
        self.execute_file_operation('FRET_BF.cpproj', "FRET明场")

    def mitochondria_nucleus_brightfield(self):
        self.execute_file_operation('Nuclei_Mit_BF.cpproj', "线粒体细胞核明场")

    def nucleus_brightfield(self):
        self.execute_file_operation('Foxo3a_BF.cpproj', "细胞核明场")

    def mitochondria_nucleus_brightfield_FRET(self):
        self.execute_file_operation('Nuclei_Mit_BF_FRET.cpproj', "线粒体细胞核明场FRET")

    def disable_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def enable_buttons(self):
        for button in self.buttons:
            button.setEnabled(True)

    def lighten_color(self, color_hex, factor):
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))

        return f"#{r:02x}{g:02x}{b:02x}"

    def show_error_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("文件打开错误")
        msg_box.setText("该文件需要 CellProffiler 进行打开")
        download_button = msg_box.addButton("访问下载", QMessageBox.ActionRole)
        ok_button = msg_box.addButton("确定", QMessageBox.AcceptRole)
        msg_box.setDefaultButton(ok_button)
        result = msg_box.exec_()
        if msg_box.clickedButton() == download_button:
            webbrowser.open("https://cellprofiler.org/releases")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PhenotypeExtractionUI()
    sys.exit(app.exec_())