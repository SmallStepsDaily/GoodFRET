import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, \
    QTextEdit, QCheckBox, QLabel

from ui.main_ui import OutputRedirector


class SegmentationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 整体垂直布局
        main_layout = QVBoxLayout()
        font = QFont()
        font.setPointSize(15)
        # 顶部输入栏布局
        top_layout = QHBoxLayout()
        self.input_folder_input = QLineEdit()
        self.input_folder_input.setPlaceholderText("输入文件夹路径")
        input_folder_button = QPushButton("选择输入文件夹")
        input_folder_button.clicked.connect(self.select_input_folder)
        self.output_folder_input = QLineEdit()
        self.output_folder_input.setPlaceholderText("输出文件夹路径")
        output_folder_button = QPushButton("选择输出文件夹")
        output_folder_button.clicked.connect(self.select_output_folder)
        top_layout.addWidget(self.input_folder_input)
        top_layout.addWidget(input_folder_button)
        top_layout.addWidget(self.output_folder_input)
        top_layout.addWidget(output_folder_button)
        main_layout.addLayout(top_layout)

        # 中间布局
        middle_layout = QHBoxLayout()

        # 中间左边文字描述
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setPlainText("""
        
        """)
        middle_layout.addWidget(self.description_text, 1)

        # 中间右边选择框和运行按钮布局
        right_layout = QVBoxLayout()
        self.fret_checkbox = QCheckBox("FRET 三通道")
        self.mitochondria_checkbox = QCheckBox("线粒体")
        self.nucleus_checkbox = QCheckBox("细胞核")
        self.cytoplasm_checkbox = QCheckBox("细胞质")
        right_layout.addWidget(self.fret_checkbox)
        right_layout.addWidget(self.mitochondria_checkbox)
        right_layout.addWidget(self.nucleus_checkbox)
        right_layout.addWidget(self.cytoplasm_checkbox)
        self.cytoplasm_checkbox.setCheckable(False)
        self.run_button = QPushButton("运行")
        self.run_button.setStyleSheet("background-color: green; color: white;")
        self.run_button.clicked.connect(self.run)
        right_layout.addWidget(self.run_button)
        middle_layout.addLayout(right_layout, 1)

        main_layout.addLayout(middle_layout)

        cmd_label = QLabel("运行输出日志 :")
        cmd_label.setFont(font)
        main_layout.addWidget(cmd_label)

        # 底部运行输出文本框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

        self.setLayout(main_layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            self.input_folder_input.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_input.setText(folder)

    def run(self):
        input_folder = self.input_folder_input.text().strip()
        output_folder = self.output_folder_input.text().strip()

        self.output_text.clear()
        self.output_text.append("开始运行==========================================>图像单细胞分割程序")

        # 重定向标准输出
        original_stdout = sys.stdout
        output_redirector = OutputRedirector(self.output_text)
        sys.stdout = output_redirector


        try:
            # 开始运行分割程序
            print("nihao")
            pass
        except Exception as e:
            self.output_text.append("运行出错==============================================>")
            self.output_text.append(str(e))
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentationUI()
    window.show()
    sys.exit(app.exec_())
