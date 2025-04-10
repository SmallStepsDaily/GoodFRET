import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QPushButton, \
    QFileDialog, QRadioButton, QButtonGroup, QLineEdit, QFrame
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt


class PhenotypeAnalysisUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 整体布局
        main_layout = QHBoxLayout()

        # 左边部分：输入表型特征文件列表
        left_layout = QVBoxLayout()
        file_list_label = QLabel("输入表型特征文件列表:")
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(2)  # 选择单个文件

        # 将 + 和 - 按钮放在同一行
        button_layout = QHBoxLayout()
        add_button = QPushButton("+ 添加文件")
        add_button.clicked.connect(self.add_file)
        remove_button = QPushButton("- 删除文件")
        remove_button.clicked.connect(self.remove_file)
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)

        left_layout.addWidget(file_list_label)
        left_layout.addWidget(self.file_list_widget)
        left_layout.addLayout(button_layout)

        # 右边部分
        right_layout = QVBoxLayout()

        # 特征融合算法选择
        algo_layout = QVBoxLayout()
        algo_label = QLabel("特征融合算法选择:")
        algo_label.setStyleSheet("font-size: 24px;")
        self.algo_button_group = QButtonGroup()
        self.pca_radio = QRadioButton("PCA算法")
        self.tsen_radio = QRadioButton("TSNE算法")
        self.lda_radio = QRadioButton("LDA算法")
        self.bayesian_radio = QRadioButton("贝叶斯网络")
        self.algo_button_group.addButton(self.pca_radio)
        self.algo_button_group.addButton(self.tsen_radio)
        self.algo_button_group.addButton(self.lda_radio)
        self.algo_button_group.addButton(self.bayesian_radio)
        self.lda_radio.setChecked(True)  # 默认选择LDA算法

        # 调整特征融合算法选择布局的边距和间距
        algo_layout.setContentsMargins(0, 0, 0, 0)
        algo_layout.setSpacing(10)

        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.pca_radio)
        algo_layout.addWidget(self.tsen_radio)
        algo_layout.addWidget(self.lda_radio)
        algo_layout.addWidget(self.bayesian_radio)

        # 输出文件路径选择和运行按钮
        output_layout = QVBoxLayout()
        output_label = QLabel("输出文件路径:")
        output_label.setStyleSheet("font-size: 24px;")
        # 将浏览按钮和输入框放在同一行
        path_layout = QHBoxLayout()
        self.output_path_entry = QLineEdit()
        browse_output_button = QPushButton("浏览")
        browse_output_button.clicked.connect(self.browse_output_file)
        path_layout.addWidget(self.output_path_entry)
        path_layout.addWidget(browse_output_button)

        run_button = QPushButton("运行")
        run_button.setStyleSheet("background-color: green;")
        run_button.clicked.connect(self.run_analysis)

        # 调整输出文件路径选择布局的边距和间距
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(5)

        output_layout.addWidget(output_label)
        output_layout.addLayout(path_layout)
        output_layout.addWidget(run_button)

        right_layout.setContentsMargins(20, 0, 0, 0)
        right_layout.addLayout(algo_layout)
        right_layout.addLayout(output_layout)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)
        self.setWindowTitle('Phenotype Analysis UI')
        self.show()

    def add_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择CSV文件", "", "CSV Files (*.csv)")
        for file_path in file_paths:
            self.file_list_widget.addItem(file_path)

    def remove_file(self):
        selected_items = self.file_list_widget.selectedItems()
        for item in selected_items:
            self.file_list_widget.takeItem(self.file_list_widget.row(item))

    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "选择输出文件路径", "", "CSV Files (*.csv)")
        if file_path:
            self.output_path_entry.setText(file_path)

    def run_analysis(self):
        file_paths = [self.file_list_widget.item(i).text() for i in range(self.file_list_widget.count())]
        algo_choice = "LDA算法" if self.lda_radio.isChecked() else "贝叶斯网络"
        output_path = self.output_path_entry.text()

        # 这里添加实际的分析代码

        result_text = f"输入文件路径: {file_paths}\n特征融合算法: {algo_choice}\n输出文件路径: {output_path}"
        print(result_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = PhenotypeAnalysisUI()
    sys.exit(app.exec_())
