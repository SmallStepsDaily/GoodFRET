import sys

import pandas as pd
from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QWidget, QHBoxLayout,
                             QPushButton, QFileDialog, QLineEdit, QFrame, QMessageBox, QApplication)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import shutil

from main import current_dir


class GaoLuWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('高璐药效函数窗口')
        self.setGeometry(300, 300, 800, 600)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 第一部分：样例图像和样例文件按钮
        top_section = QHBoxLayout()
        top_section.setSpacing(20)

        # 样例图像
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        self.image_frame.setMinimumHeight(200)

        image_label = QLabel("样例图像")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("font-weight: bold; padding: 10px;")

        self.sample_image = QLabel()
        self.sample_image.setAlignment(Qt.AlignCenter)
        self.sample_image.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 3px;")

        image_layout = QVBoxLayout()
        image_layout.addWidget(image_label)
        image_layout.addWidget(self.sample_image, 1)

        self.image_frame.setLayout(image_layout)
        top_section.addWidget(self.image_frame, 1)

        # 样例文件按钮区域
        sample_file_frame = QFrame()
        sample_file_frame.setFrameShape(QFrame.StyledPanel)
        sample_file_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")

        sample_file_layout = QVBoxLayout()

        info_label = QLabel("样例文件可另存点击复制")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; padding: 10px;")

        self.sample_button = QPushButton("样例文件 (CSV)")
        self.sample_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a6ebd;
            }
        """)
        self.sample_button.setMinimumHeight(40)
        self.sample_button.clicked.connect(self.copy_sample_file)

        sample_file_layout.addWidget(info_label)
        sample_file_layout.addWidget(self.sample_button)
        sample_file_layout.addStretch()

        sample_file_frame.setLayout(sample_file_layout)
        top_section.addWidget(sample_file_frame, 1)

        main_layout.addLayout(top_section)

        # 第二部分：输入文件路径
        file_path_frame = QFrame()
        file_path_frame.setFrameShape(QFrame.StyledPanel)
        file_path_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px; padding: 15px;")

        file_path_layout = QHBoxLayout()

        file_label = QLabel("选择CSV文件:")
        file_label.setMinimumWidth(100)
        file_label.setStyleSheet("font-weight: bold;")

        self.file_path_entry = QLineEdit()
        self.file_path_entry.setReadOnly(True)
        self.file_path_entry.setPlaceholderText("请选择CSV格式文件")

        browse_button = QPushButton("浏览")
        browse_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        browse_button.clicked.connect(self.browse_file)

        file_path_layout.addWidget(file_label)
        file_path_layout.addWidget(self.file_path_entry, 1)
        file_path_layout.addWidget(browse_button)

        file_path_frame.setLayout(file_path_layout)
        main_layout.addWidget(file_path_frame)

        # 第三部分：运行按钮
        run_button = QPushButton("运行")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-size: 18px;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F4511E;
            }
            QPushButton:pressed {
                background-color: #E64A19;
            }
        """)
        run_button.setMinimumHeight(50)
        run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(run_button)

        self.setLayout(main_layout)

        # 加载样例图像
        self.load_sample_image()

    def load_sample_image(self):
        # 加载实际的样例图像
        image_path = current_dir + "/data/image/药效文件样例.png"

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.sample_image.setPixmap(pixmap.scaled(
                    self.sample_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                self.sample_image.setText("无法加载图像")
        else:
            self.sample_image.setText("图像文件不存在:\n" + image_path)

    def resizeEvent(self, event):
        # 窗口大小变化时调整图像大小
        if not self.sample_image.pixmap() or self.sample_image.pixmap().isNull():
            self.load_sample_image()
        else:
            self.sample_image.setPixmap(self.sample_image.pixmap().scaled(
                self.sample_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        super().resizeEvent(event)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.file_path_entry.setText(file_path)

    def copy_sample_file(self):
        # 复制实际的样例CSV文件
        sample_file_path = current_dir + "/data/example/sample_drug_efficacy_file.csv"

        if not os.path.exists(sample_file_path):
            QMessageBox.critical(self, "错误", f"样例文件不存在:\n{sample_file_path}")
            return

        # 设置默认文件名和路径
        default_dir = os.path.expanduser("~")  # 默认保存到用户主目录
        default_file_name = "药物评价.csv"
        default_path = os.path.join(default_dir, default_file_name)

        # 提示用户保存文件，设置默认文件名
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存样例文件", default_path, "CSV Files (*.csv)"
        )

        if save_path:
            try:
                # 复制文件到用户选择的位置
                shutil.copy2(sample_file_path, save_path)
                QMessageBox.information(self, "成功", f"样例文件已保存到:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存文件时出错:\n{str(e)}")

    def run_analysis(self):
        file_path = self.file_path_entry.text()
        self.folder_path = os.path.dirname(file_path)
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择CSV文件")
            return

        if not file_path.lower().endswith('.csv'):
            QMessageBox.warning(self, "警告", "请选择CSV格式的文件")
            return

        # 这里添加实际的分析逻辑
        QMessageBox.information(self, "开始分析", f"正在分析文件:\n{file_path}")
        # 后续处理代码...

        from analysis.pharmacodynamics.gl import gaolu_function
        data = pd.read_csv(file_path, encoding="utf-8")
        result_str = gaolu_function(data)
        # 保存文本结果
        text_path = os.path.join(self.folder_path, "药效值.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(result_str)

        QMessageBox.information(self, "成功", f"药效值已经计算完成！")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = GaoLuWindow()
    ui.show()
    sys.exit(app.exec_())