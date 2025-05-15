import base64
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QRadioButton, QButtonGroup, QComboBox,
                             QTextEdit, QFrame, QMessageBox, QSplitter, QScrollArea, QSizePolicy)
from PyQt5.QtGui import QCursor, QPixmap, QImage, QPalette
from PyQt5.QtCore import Qt, QSize


class FRETAnalysisUI(QWidget):
    def __init__(self):
        super().__init__()
        # 选择的处理方法
        self.dim_method = None
        self.data_df = None
        self.file_path = ''
        self.save_dir = ''
        self.current_pixmap = None
        self.current_result_str = ''
        self.auto_fit_mode = True  # 自动适应模式
        self.max_scale_factor = 3.0  # 最大缩放比例
        self.scale_factor = 1.0  # 当前缩放比例
        self.initUI()

    def initUI(self):
        # 整体布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 第一部分：输入文件路径
        file_path_layout = QHBoxLayout()
        file_label = QLabel("请输入csv文件路径:")
        file_label.setMinimumWidth(120)
        self.file_path_entry = QLineEdit()
        self.file_path_entry.setReadOnly(True)  # 设置为只读，防止手动输入错误路径
        browse_button = QPushButton("浏览")
        browse_button.setFixedWidth(80)
        browse_button.clicked.connect(self.browse_file)

        file_path_layout.addWidget(file_label)
        file_path_layout.addWidget(self.file_path_entry)
        file_path_layout.addWidget(browse_button)
        main_layout.addLayout(file_path_layout)

        # 第二部分：参数选择
        param_layout = QHBoxLayout()
        param_layout.setSpacing(20)

        # FRET特征选择
        fret_layout = QVBoxLayout()
        fret_layout.setSpacing(10)
        fret_label = QLabel("FRET特征选择:")
        fret_label.setStyleSheet("font-weight: bold;")

        self.single_feature_radio = QRadioButton("单一特征分析")
        self.single_feature_radio.toggled.connect(self.toggle_feature_choice)

        self.feature_combobox = QComboBox()
        self.feature_combobox.setEnabled(False)
        self.feature_combobox.setMinimumHeight(30)

        self.all_feature_radio = QRadioButton("整体特征分析")
        self.all_feature_radio.toggled.connect(self.toggle_feature_choice)
        self.all_feature_radio.setChecked(False)
        # TODO 还有待完善该功能算法
        self.all_feature_radio.setEnabled(False)

        fret_layout.addWidget(fret_label)
        fret_layout.addWidget(self.single_feature_radio)
        fret_layout.addWidget(self.feature_combobox)
        fret_layout.addWidget(self.all_feature_radio)
        fret_layout.addStretch()

        # 特征降维算法选择
        dim_layout = QVBoxLayout()
        dim_layout.setSpacing(10)
        dim_label = QLabel("特征降维算法选择:")
        dim_label.setStyleSheet("font-weight: bold;")

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
        dim_layout.addStretch()

        param_layout.addLayout(fret_layout)
        param_layout.addLayout(dim_layout)
        main_layout.addLayout(param_layout)

        # 第三部分：结果展示 - 使用QSplitter实现可调整大小的区域
        splitter = QSplitter(Qt.Horizontal)

        # 图像输出区域
        self.image_frame = QFrame()
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")

        self.image_layout = QVBoxLayout()

        # 图像控制按钮
        control_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.auto_fit_btn = QPushButton("自适应")

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.auto_fit_btn.clicked.connect(self.auto_fit)

        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.auto_fit_btn)

        image_label = QLabel("图像输出区域")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("font-weight: bold; padding: 5px;")

        self.image_layout.addLayout(control_layout)
        self.image_layout.addWidget(image_label)

        # 使用QScrollArea来显示图像，支持滚动
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setBackgroundRole(QPalette.Light)

        # 用于显示图像的QLabel
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 3px;")
        self.image_display.setMinimumSize(100, 100)  # 设置最小尺寸

        self.scroll_area.setWidget(self.image_display)
        self.image_layout.addWidget(self.scroll_area, 1)

        self.image_frame.setLayout(self.image_layout)
        splitter.addWidget(self.image_frame)

        # 文本输出区域
        text_frame = QFrame()
        text_frame.setFrameShape(QFrame.StyledPanel)
        text_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")

        text_label = QLabel("运行结果输出区域")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("font-weight: bold; padding: 5px;")

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setCursor(QCursor())
        self.text_output.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 3px;")

        text_layout = QVBoxLayout()
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_output, 1)

        text_frame.setLayout(text_layout)
        splitter.addWidget(text_frame)

        # 设置分割器初始大小比例
        splitter.setSizes([450, 450])

        main_layout.addWidget(splitter, 1)

        # 第四部分：运行按钮
        run_button = QPushButton("运行分析")
        run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        run_button.setMinimumHeight(40)
        run_button.clicked.connect(self.run_analysis)
        main_layout.addWidget(run_button)

        # 第五部分：保存路径选择（现在放在运行按钮下方）
        save_path_layout = QHBoxLayout()
        save_label = QLabel("保存文件夹路径:")
        save_label.setMinimumWidth(120)

        self.save_path_entry = QLineEdit()
        self.save_path_entry.setReadOnly(True)

        browse_save_button = QPushButton("浏览")
        browse_save_button.setFixedWidth(80)
        browse_save_button.clicked.connect(self.browse_save_dir)

        save_button = QPushButton("保存结果")
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0a6ebd;
            }
        """)
        save_button.setFixedWidth(100)
        save_button.clicked.connect(self.save_results)

        save_path_layout.addWidget(save_label)
        save_path_layout.addWidget(self.save_path_entry)
        save_path_layout.addWidget(browse_save_button)
        save_path_layout.addWidget(save_button)

        main_layout.addLayout(save_path_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('FRET数据分析工具')
        self.setGeometry(300, 300, 900, 700)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.file_path = file_path
            self.file_path_entry.setText(file_path)

            # 检查文件扩展名
            if not file_path.lower().endswith('.csv'):
                QMessageBox.warning(self, "文件格式错误", "请选择CSV格式的文件!")
                self.file_path_entry.clear()
                self.file_path = ''
                return

            try:
                columns = self.read_pd(file_path)
                self.feature_combobox.clear()
                self.feature_combobox.addItems(columns)
                self.text_output.append(f"成功加载文件: {file_path}\n找到 {len(columns)} 个特征列")
            except Exception as e:
                QMessageBox.critical(self, "文件读取错误", f"无法读取文件: {str(e)}")
                self.file_path_entry.clear()
                self.file_path = ''

    def read_pd(self, file_path):
        # 添加文件读取进度提示
        self.text_output.append("正在读取文件...")
        self.data_df = pd.read_csv(file_path)

        # 过滤不需要的列
        columns = [
            col for col in self.data_df.columns
            if not (col.startswith('Metadata_') or col in ['ObjectNumber', 'ImageNumber'])
        ]

        return columns

    def toggle_feature_choice(self):
        self.feature_combobox.setEnabled(self.single_feature_radio.isChecked())

    def browse_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择保存文件夹", "", QFileDialog.ShowDirsOnly
        )

        if dir_path:
            self.save_dir = dir_path
            self.save_path_entry.setText(dir_path)

    def save_results(self):
        if not self.save_dir:
            QMessageBox.warning(self, "操作错误", "请先选择保存文件夹!")
            return

        if not self.current_pixmap or not self.current_result_str:
            QMessageBox.warning(self, "操作错误", "没有可保存的结果!")
            return

        try:
            def save(image_name, csv_name):
                # 保存图像
                image_path = os.path.join(self.save_dir, image_name)
                self.current_pixmap.save(image_path, "PNG")

                # 保存文本结果
                text_path = os.path.join(self.save_dir, csv_name)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(self.current_result_str)

                QMessageBox.information(self, "保存成功", f"结果已成功保存到:\n{image_path}\n{text_path}")
            if self.dim_method == "标准化降维":
                save("标准化统计.png", "SD标准化计算结果.txt")
            elif self.dim_method == "JS散度降维":
                save("概率密度.png", "JS散度计算结果.txt")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存结果时发生错误: {str(e)}")

    def run_analysis(self):
        file_path = self.file_path_entry.text()

        if not file_path:
            QMessageBox.warning(self, "操作错误", "请先选择CSV文件!")
            return

        if self.data_df is None:
            QMessageBox.warning(self, "操作错误", "文件未成功加载，请检查文件格式和内容!")
            return

        feature_choice = "单一特征分析" if self.single_feature_radio.isChecked() else "整体特征分析"

        if self.standard_dim_radio.isChecked():
            self.dim_method = "标准化降维"
        elif self.js_dim_radio.isChecked():
            self.dim_method = "JS散度降维"
        else:
            self.dim_method = "绝对值降维"

        # 构建结果文本
        result_text = f"""
        ====================================
        分析参数:
        - 文件路径: {file_path}
        - 特征选择: {feature_choice}
        - 降维方法: {self.dim_method}

        数据概览:
        - 样本数: {len(self.data_df)}
        - 特征数: {len(self.data_df.columns)}
        ====================================
        开始分析...
        """
        self.text_output.setText(result_text)

        if self.standard_dim_radio.isChecked():
            from analysis.fret.standard_deviation import SD
            sd_model = SD(self.data_df)
            if self.single_feature_radio.isChecked():
                selected_feature = self.feature_combobox.currentText()
                self.text_output.append(f"正在分析特征: {selected_feature}")

                try:
                    # 传递选中的特征作为参数
                    values, result_str, image = sd_model.start(feature_name=selected_feature)
                    self.current_result_str = result_str
                    self.text_output.append(result_str)

                    # 更新图像显示
                    self._display_image(image)
                except Exception as e:
                    self.text_output.append(f"分析过程中发生错误: {str(e)}")


        elif self.js_dim_radio.isChecked():
            from analysis.fret.js import JSDivergence  # 修正包名拼写错误
            js_model = JSDivergence(self.data_df)
            # TODO 整体分析和单一特征分析
            if self.single_feature_radio.isChecked():
                selected_feature = self.feature_combobox.currentText()
                self.text_output.append(f"正在分析特征: {selected_feature}")

                try:
                    # 传递选中的特征作为参数
                    values, result_str, image = js_model.start(feature_name=selected_feature)
                    self.current_result_str = result_str
                    self.text_output.append(result_str)

                    # 更新图像显示
                    self._display_image(image)
                except Exception as e:
                    self.text_output.append(f"分析过程中发生错误: {str(e)}")
        else:
            pass

    def _display_image(self, image):
        """显示图像并支持动态调整大小"""
        if not image:
            self.text_output.append("警告: 没有生成图像")
            self.image_display.setText("无图像")
            return

        try:
            if isinstance(image, str):
                # 处理base64编码的字符串
                try:
                    # 移除可能存在的Data URI前缀
                    if image.startswith('data:image/png;base64,'):
                        image = image.split(',', 1)[1]

                    # 解码base64字符串为字节
                    image_data = base64.b64decode(image)

                    # 创建QPixmap
                    pixmap = QPixmap()
                    if not pixmap.loadFromData(image_data, "PNG"):
                        self.text_output.append("错误: 无法加载PNG图像数据")
                        self.image_display.setText("无法加载图像")
                        return
                except Exception as e:
                    self.text_output.append(f"解码图像数据时发生错误: {str(e)}")
                    self.image_display.setText("解码图像失败")
                    return
            elif isinstance(image, QPixmap):
                pixmap = image
            elif isinstance(image, QImage):
                pixmap = QPixmap.fromImage(image)
            else:
                self.text_output.append(f"错误: 不支持的图像类型 ({type(image).__name__})")
                self.image_display.setText("不支持的图像类型")
                return

            # 保存当前图像引用
            self.current_pixmap = pixmap
            self.scale_factor = 1.0  # 重置缩放比例

            # 显示图像
            self._update_image_display()
            self.text_output.append("图像已成功显示")

        except Exception as e:
            self.text_output.append(f"显示图像时发生错误: {str(e)}")
            self.image_display.setText("显示图像失败")

    def _update_image_display(self):
        """更新图像显示，根据当前模式调整图像大小"""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return

        if self.auto_fit_mode:
            # 自动适应模式：缩放图像以适应显示区域，但不超过原始大小
            display_size = self.scroll_area.viewport().size()
            original_size = self.current_pixmap.size()

            # 计算适应显示区域的缩放比例
            scale_width = min(1.0, display_size.width() / original_size.width())
            scale_height = min(1.0, display_size.height() / original_size.height())
            scale_factor = min(scale_width, scale_height)

            # 缩放图像
            scaled_pixmap = self.current_pixmap.scaled(
                original_size * scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_display.setPixmap(scaled_pixmap)
            self.image_display.adjustSize()
        else:
            # 缩放模式：根据当前缩放因子显示图像
            scaled_pixmap = self.current_pixmap.scaled(
                self.current_pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
            self.image_display.adjustSize()

    def resizeEvent(self, event):
        """处理窗口大小变化事件，更新图像显示"""
        if self.current_pixmap and not self.current_pixmap.isNull() and self.auto_fit_mode:
            self._update_image_display()

        # 调用基类的resizeEvent方法
        super(FRETAnalysisUI, self).resizeEvent(event)

    def zoom_in(self):
        """放大图像"""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return

        self.auto_fit_mode = False
        self.scale_factor = min(self.scale_factor * 1.2, self.max_scale_factor)
        self._update_image_display()

    def zoom_out(self):
        """缩小图像"""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return

        self.auto_fit_mode = False
        self.scale_factor = max(self.scale_factor / 1.2, 0.1)
        self._update_image_display()

    def auto_fit(self):
        """自动适应图像大小"""
        self.auto_fit_mode = True
        self._update_image_display()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格使界面更现代
    ui = FRETAnalysisUI()
    ui.show()
    sys.exit(app.exec_())