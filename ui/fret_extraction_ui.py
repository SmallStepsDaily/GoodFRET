import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, \
    QLabel, QTextEdit, QRadioButton, QButtonGroup, QGridLayout, QCheckBox

from ui import TextUpdateHandler


class FRETExtractionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.running = False
        self.run_button = None
        self.stop_button = None
        self.initUI()

    def initUI(self):
        # 创建主布局
        main_layout = QVBoxLayout()

        # 输入文件夹路径部分
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("输入文件夹路径")
        select_folder_button = QPushButton("选择文件夹")
        select_folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(select_folder_button)

        # 中间四列布局，使用 QGridLayout
        middle_layout = QGridLayout()

        # 设置中间四列等比例分布
        for i in range(4):
            middle_layout.setColumnStretch(i * 2, 1)
            middle_layout.setColumnStretch(i * 2 + 1, 1)

        # FRET 图像参数部分
        fret_param_label = QLabel("FRET 图像参数")
        fret_param_label.setStyleSheet("font-size: 24px;")
        middle_layout.addWidget(fret_param_label, 0, 0, 1, 2)

        # AA 曝光时间
        aa_label = QLabel("AA 曝光时间 (ms):")
        aa_label.setStyleSheet("font-size: 18px;")
        self.aa_input = QLineEdit()
        self.aa_input.setText("300")
        middle_layout.addWidget(aa_label, 1, 0)
        middle_layout.addWidget(self.aa_input, 1, 1)

        # DD 曝光时间
        dd_label = QLabel("DD 曝光时间 (ms):")
        dd_label.setStyleSheet("font-size: 18px;")
        self.dd_input = QLineEdit()
        self.dd_input.setText("300")
        middle_layout.addWidget(dd_label, 2, 0)
        middle_layout.addWidget(self.dd_input, 2, 1)

        # DA 曝光时间
        da_label = QLabel("DA 曝光时间 (ms):")
        da_label.setStyleSheet("font-size: 18px;")
        self.da_input = QLineEdit()
        self.da_input.setText("300")
        middle_layout.addWidget(da_label, 3, 0)
        middle_layout.addWidget(self.da_input, 3, 1)

        # RC/ED 筛选参数部分
        rc_ed_param_label = QLabel("RC/ED 筛选参数")
        rc_ed_param_label.setStyleSheet("font-size: 24px;")
        middle_layout.addWidget(rc_ed_param_label, 0, 2, 1, 2)

        # rc 最小值
        rc_min_label = QLabel("rc 最小值:")
        rc_min_label.setStyleSheet("font-size: 18px;")
        self.rc_min_input = QLineEdit()
        self.rc_min_input.setText("0.0")
        middle_layout.addWidget(rc_min_label, 1, 2)
        middle_layout.addWidget(self.rc_min_input, 1, 3)

        # rc 最大值
        rc_max_label = QLabel("rc 最大值:")
        rc_max_label.setStyleSheet("font-size: 18px;")
        self.rc_max_input = QLineEdit()
        self.rc_max_input.setText("2.5")
        middle_layout.addWidget(rc_max_label, 2, 2)
        middle_layout.addWidget(self.rc_max_input, 2, 3)

        # ed 最小值
        ed_min_label = QLabel("ed 最小值:")
        ed_min_label.setStyleSheet("font-size: 18px;")
        self.ed_min_input = QLineEdit()
        self.ed_min_input.setText("0.0")
        middle_layout.addWidget(ed_min_label, 3, 2)
        middle_layout.addWidget(self.ed_min_input, 3, 3)

        # ed 最大值
        ed_max_label = QLabel("ed 最大值:")
        ed_max_label.setStyleSheet("font-size: 18px;")
        self.ed_max_input = QLineEdit()
        self.ed_max_input.setText("1.0")
        middle_layout.addWidget(ed_max_label, 4, 2)
        middle_layout.addWidget(self.ed_max_input, 4, 3)

        # 靶点选择部分
        target_selection_label = QLabel("靶点选择")
        target_selection_label.setStyleSheet("font-size: 24px;")
        middle_layout.addWidget(target_selection_label, 0, 4, 1, 2)

        self.target_egfr_grb2 = QRadioButton("egfr_grb2")
        self.target_bax_bak = QRadioButton("bax_bak")
        self.target_button_group = QButtonGroup()
        self.target_button_group.addButton(self.target_egfr_grb2)
        self.target_button_group.addButton(self.target_bax_bak)
        self.target_egfr_grb2.setChecked(True)

        middle_layout.addWidget(self.target_egfr_grb2, 1, 4)
        middle_layout.addWidget(self.target_bax_bak, 2, 4)

        # 特征尺度选择部分
        feature_scale_label = QLabel("特征尺度")
        feature_scale_label.setStyleSheet("font-size: 24px;")
        middle_layout.addWidget(feature_scale_label, 0, 5, 1, 2)

        self.extract_single_cell = QRadioButton("提取单细胞特征")
        self.extract_whole_image = QRadioButton("提取整图像特征")
        self.feature_button_group = QButtonGroup()
        self.feature_button_group.addButton(self.extract_single_cell)
        self.feature_button_group.addButton(self.extract_whole_image)

        self.extract_single_cell.setChecked(True)

        middle_layout.addWidget(self.extract_single_cell, 1, 5)
        middle_layout.addWidget(self.extract_whole_image, 2, 5)

        # 特征类型选择部分
        feature_selection_label = QLabel("特征选择")
        feature_selection_label.setStyleSheet("font-size: 24px;")
        middle_layout.addWidget(feature_selection_label, 0, 6, 1, 2)

        self.Ed_feature = QCheckBox("FRET-Ed特征")
        self.Rc_feature = QCheckBox("FRET-Rc特征")
        self.Fp_feature = QCheckBox("FRET-Fp特征")
        self.Rc_Ed_feature = QCheckBox("统计Rc-Ed关系")
        self.Ed_feature.setChecked(True)
        self.Ed_feature.setEnabled(False)
        self.Rc_feature.setChecked(True)
        self.Fp_feature.setChecked(True)
        self.Rc_Ed_feature.setChecked(False)
        middle_layout.addWidget(self.Ed_feature, 1, 6)
        middle_layout.addWidget(self.Rc_feature, 2, 6)
        middle_layout.addWidget(self.Fp_feature, 3, 6)
        middle_layout.addWidget(self.Rc_Ed_feature, 4, 6)
        # 运行日志标签和命令输出文本框的水平布局
        log_layout = QHBoxLayout()
        log_label = QLabel("运行日志")
        log_label.setStyleSheet("font-size: 24px;")
        self.output_text = QTextEdit()
        self.text_handler = TextUpdateHandler(self.output_text)
        self.output_text.setReadOnly(True)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.output_text)

        # 添加运行和终止按钮
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("运行")
        self.run_button.setStyleSheet("background-color: green; color: white;")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("终止")
        self.stop_button.setStyleSheet("background-color: red; color: white;")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)

        # 添加各部分到主布局
        main_layout.addLayout(folder_layout)
        main_layout.addLayout(middle_layout)
        main_layout.addLayout(log_layout)
        main_layout.addLayout(button_layout)

        # 设置比例
        main_layout.setStretchFactor(folder_layout, 2)
        main_layout.setStretchFactor(middle_layout, 3)
        main_layout.setStretchFactor(log_layout, 5)
        main_layout.setStretchFactor(button_layout, 1)

        self.setLayout(main_layout)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            self.folder_input.setText(folder)

    def run(self):
        if self.running:
            return
        self.running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_event.clear()  # 清除停止事件
        input_folder = self.folder_input.text().strip()
        self.output_text.clear()

        self.output_text.append("输入文件路径为: " + str(input_folder))
        self.output_text.append("开始运行==========================================>FRET特征提取程序")

        try:
            # 为运行操作创建一个新线程
            threading.Thread(target=self._run_fret_extraction, args=(input_folder, self.text_handler)).start()
        except Exception as e:
            self.output_text.append("运行出错==============================================>" + str(e))
            self.running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)


    def _run_fret_extraction(self, input_folder, output_redirector):
        # 保证text输出安全，添加信号机制

        try:
            target_checked_button = self.target_button_group.checkedButton()
            feature_checked_button = self.feature_button_group.checkedButton()
            fret_target_name = ''
            if target_checked_button == self.target_bax_bak and feature_checked_button == self.extract_single_cell:
                fret_target_name = 'bax_bak'
            if target_checked_button == self.target_egfr_grb2 and feature_checked_button == self.extract_single_cell:
                fret_target_name = 'egfr_grb2'
            need_Ed = False
            need_Rc = False
            need_Fp = False
            need_Rc_Ed = False
            if self.Ed_feature.isChecked():
                need_Ed = True
            if self.Rc_feature.isChecked():
                need_Rc = True
            if self.Fp_feature.isChecked():
                need_Fp = True
            if self.Rc_Ed_feature.isChecked():
                need_Rc_Ed = True
            from extracting.compute import FRETComputer
            from batch.processing import BatchProcessing
            # 参数Ed提取参数 验证批处理流程
            def process(image_set_path, fret_model):
                #############################
                # EGFR-FRET分析流程
                #############################
                # 进行分割流程
                return fret_model.start(image_set_path)

            fret = FRETComputer(fret_target_name,
                                rc_min=float(self.rc_min_input.text()),
                                rc_max=float(self.rc_max_input.text()),
                                ed_min=float(self.ed_min_input.text()),
                                ed_max=float(self.ed_max_input.text()),
                                expose_times=(int(self.aa_input.text()),
                                              int(self.dd_input.text()),
                                              int(self.da_input.text())),
                                need_Ed=need_Ed,
                                need_Rc=need_Rc,
                                need_Fp=need_Fp,
                                need_Rc_Ed=need_Rc_Ed,
                                output_redirector=output_redirector)
            batch = BatchProcessing(input_folder, stop_event=self.stop_event)
            batch.start(process, fret)
            self.output_text.append("运行完成==============================================>FRET特征提取程序")
        except Exception as e:
            print("运行出错==============================================>" + str(e))
            self.output_text.append("运行出错==============================================>" + str(e))
        finally:
            self.running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def stop(self):
        self.running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_event.set()  # 设置停止事件
        self.output_text.append("运行已终止==============================================>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FRETExtractionUI()
    sys.exit(app.exec_())
