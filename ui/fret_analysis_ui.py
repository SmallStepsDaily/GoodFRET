import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, \
    QFileDialog, \
    QRadioButton, QButtonGroup, QComboBox, QTextEdit, QFrame
from PyQt5.QtGui import QCursor


class FRETAnalysisUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 整体布局
        main_layout = QVBoxLayout()

        # 第一部分：输入文件路径
        file_path_layout = QHBoxLayout()
        file_label = QLabel("请输入csv文件路径:")
        self.file_path_entry = QLineEdit()
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(self.browse_file)

        file_path_layout.addWidget(file_label)
        file_path_layout.addWidget(self.file_path_entry)
        file_path_layout.addWidget(browse_button)
        main_layout.addLayout(file_path_layout)

        # 第二部分：参数选择
        param_layout = QHBoxLayout()

        # FRET特征选择
        fret_layout = QVBoxLayout()
        fret_label = QLabel("FRET特征选择:")
        self.single_feature_radio = QRadioButton("单一特征分析")
        self.single_feature_radio.toggled.connect(self.toggle_feature_choice)
        self.all_feature_radio = QRadioButton("整体特征分析")
        self.all_feature_radio.toggled.connect(self.toggle_feature_choice)
        self.feature_combobox = QComboBox()
        self.feature_combobox.setEnabled(False)

        fret_layout.addWidget(fret_label)
        fret_layout.addWidget(self.single_feature_radio)
        fret_layout.addWidget(self.feature_combobox)
        fret_layout.addWidget(self.all_feature_radio)

        # 特征降维算法选择
        dim_layout = QVBoxLayout()
        dim_label = QLabel("特征降维算法选择:")
        self.dim_button_group = QButtonGroup()
        self.standard_dim_radio = QRadioButton("标准化降维")
        self.js_dim_radio = QRadioButton("JS散度降维")
        self.js_dim_radio.setChecked(True)
        self.abs_dim_radio = QRadioButton("绝对值降维")

        self.dim_button_group.addButton(self.standard_dim_radio)
        self.dim_button_group.addButton(self.js_dim_radio)
        self.dim_button_group.addButton(self.abs_dim_radio)

        dim_layout.addWidget(dim_label)
        dim_layout.addWidget(self.standard_dim_radio)
        dim_layout.addWidget(self.js_dim_radio)
        dim_layout.addWidget(self.abs_dim_radio)

        param_layout.addLayout(fret_layout)
        param_layout.addLayout(dim_layout)
        main_layout.addLayout(param_layout)

        # 第三部分：结果展示
        result_layout = QHBoxLayout()

        # 图像输出区域
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_label = QLabel("图像输出区域")
        image_layout = QVBoxLayout()
        image_layout.addWidget(image_label)
        image_frame.setLayout(image_layout)

        # 文本输出区域
        text_frame = QFrame()
        text_frame.setFrameShape(QFrame.StyledPanel)
        text_label = QLabel("运行结果输出区域")
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setCursor(QCursor())
        text_layout = QVBoxLayout()
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_output)
        text_frame.setLayout(text_layout)

        result_layout.addWidget(image_frame, 1)
        result_layout.addWidget(text_frame, 1)
        main_layout.addLayout(result_layout)

        # 运行按钮
        run_button = QPushButton("运行")
        run_button.setStyleSheet("background-color: green;")
        run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(run_button)

        self.setLayout(main_layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path_entry.setText(file_path)
            try:
                df = pd.read_csv(file_path)
                columns = df.columns.tolist()
                self.feature_combobox.clear()
                self.feature_combobox.addItems(columns)
            except Exception as e:
                print(f"读取文件出错: {e}")

    def toggle_feature_choice(self):
        if self.single_feature_radio.isChecked():
            self.feature_combobox.setEnabled(True)
            self.all_feature_radio.setChecked(False)
        else:
            self.feature_combobox.setEnabled(False)
            self.single_feature_radio.setChecked(False)

    def run_analysis(self):
        file_path = self.file_path_entry.text()
        feature_choice = "单一特征分析" if self.single_feature_radio.isChecked() else "整体特征分析"
        if self.standard_dim_radio.isChecked():
            dim_method = "标准化降维"
        elif self.js_dim_radio.isChecked():
            dim_method = "JS散度降维"
        else:
            dim_method = "绝对值降维"

        result_text = f"文件路径: {file_path}\n特征选择: {feature_choice}\n降维方法: {dim_method}"
        self.text_output.setText(result_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = FRETAnalysisUI()
    sys.exit(app.exec_())
