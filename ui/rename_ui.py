import sys

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, QListWidget, \
    QInputDialog, QTextEdit, QLabel
from PyQt5.QtCore import Qt

from batch.processing import BatchProcessing
from rename.update import update_file_name

class RenameUI(QWidget):
    def __init__(self):
        super().__init__()
        # 设置 leftlist 的初始值
        self.leftlist = ["AA.tif", "DA.tif", "DD.tif"]
        self.rightlist = []
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        # 创建 QFont 对象并设置字体大小
        font = QFont()
        font.setPointSize(15)
        # 顶部输入框和选择文件夹按钮
        top_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("请输入文件路径")
        top_layout.addWidget(self.file_path_input, stretch=1)
        self.select_folder_button = QPushButton("选择文件夹")
        self.select_folder_button.clicked.connect(self.select_folder)
        top_layout.addWidget(self.select_folder_button)
        main_layout.addLayout(top_layout)

        # 中部布局
        middle_layout = QHBoxLayout()

        # 中左序列框和 + - 按钮
        left_middle_layout = QVBoxLayout()
        label = QPushButton("检测图像完整性名称列表")
        label.setFont(font)
        label.setFlat(True)
        label.setStyleSheet("background-color: transparent;")
        label.setDisabled(True)
        left_middle_layout.addWidget(label)
        self.check_list = QListWidget()
        # 提前加载 leftlist 的初始值到 QListWidget 中
        for item in self.leftlist:
            self.check_list.addItem(item)
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("添加")
        self.add_button.clicked.connect(self.add_item_to_check_list)
        self.remove_button = QPushButton("删除")
        self.remove_button.clicked.connect(self.remove_item_from_check_list)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        left_middle_layout.addWidget(self.check_list)
        left_middle_layout.addLayout(button_layout)
        middle_layout.addLayout(left_middle_layout)

        # 中右序列框和添加删除按钮
        right_middle_layout = QVBoxLayout()
        right_label = QPushButton("image_开头文件修改名称列表")
        right_label.setFont(font)
        right_label.setFlat(True)
        right_label.setStyleSheet("background-color: transparent;")
        right_label.setDisabled(True)
        right_middle_layout.addWidget(right_label)
        self.modify_list = QListWidget()
        right_button_layout = QHBoxLayout()
        self.add_to_modify_button = QPushButton("添加")
        self.add_to_modify_button.clicked.connect(self.add_item_to_modify_list)
        self.remove_from_modify_button = QPushButton("删除")
        self.remove_from_modify_button.clicked.connect(self.remove_item_from_modify_list)
        right_button_layout.addWidget(self.add_to_modify_button)
        right_button_layout.addWidget(self.remove_from_modify_button)
        right_middle_layout.addWidget(self.modify_list)
        right_middle_layout.addLayout(right_button_layout)
        middle_layout.addLayout(right_middle_layout)

        main_layout.addLayout(middle_layout)

        cmd_label = QLabel("运行输出日志 :")
        cmd_label.setFont(font)
        main_layout.addWidget(cmd_label)
        # 底部运行文本框
        self.run_text_box = QTextEdit()
        self.run_text_box.setReadOnly(True)
        main_layout.addWidget(self.run_text_box)

        # 添加运行按钮并设置样式
        self.run_button = QPushButton("运行")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #003d80;
            }
        """)
        self.run_button.clicked.connect(self.on_run_button_clicked)
        main_layout.addWidget(self.run_button)

        self.setLayout(main_layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.file_path_input.setText(folder_path)

    def add_item_to_check_list(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("添加文件")
        dialog.setLabelText("请输入文件名称")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        if dialog.exec_() == QInputDialog.Accepted:
            text = dialog.textValue()
            if text:
                self.check_list.addItem(text)
                self.leftlist.append(text)

    def remove_item_from_check_list(self):
        selected_items = self.check_list.selectedItems()
        for item in selected_items:
            row = self.check_list.row(item)
            self.check_list.takeItem(row)
            if item.text() in self.leftlist:
                self.leftlist.remove(item.text())

    def add_item_to_modify_list(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("添加文件")
        dialog.setLabelText("请输入文件名称")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        if dialog.exec_() == QInputDialog.Accepted:
            text = dialog.textValue()
            if text:
                self.modify_list.addItem(text)
                self.rightlist.append(text)

    def remove_item_from_modify_list(self):
        selected_items = self.modify_list.selectedItems()
        for item in selected_items:
            row = self.modify_list.row(item)
            self.modify_list.takeItem(row)
            if item.text() in self.rightlist:
                self.rightlist.remove(item.text())

    def on_run_button_clicked(self):
        # 获取文件路径输入框中的文本并记录为 file_path
        file_path = self.file_path_input.text()
        self.run_text_box.clear()
        self.run_text_box.append(f"获取到的文件路径参数: {file_path}")
        self.run_text_box.append("开始运行==========================================>图像名称修改程序")

        try:
            batch = BatchProcessing(file_path)
            batch.start(update_file_name, self.leftlist, self.rightlist, self.run_text_box)
        except Exception as e:
            self.run_text_box.append("运行出错==============================================>")
            self.run_text_box.append(str(e))
