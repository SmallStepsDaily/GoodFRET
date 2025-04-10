import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QFileDialog, \
    QTextEdit, QRadioButton, QLabel, QButtonGroup
import threading
from PyQt5.QtCore import pyqtSignal, QObject

from batch.processing import BatchProcessing
from ui.main_ui import OutputRedirector


class SegmentationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.running = False
        self.stop_event = threading.Event()  # 添加 stop_event
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
        top_layout.addWidget(self.input_folder_input)
        top_layout.addWidget(input_folder_button)
        main_layout.addLayout(top_layout)

        # 中间布局
        middle_layout = QHBoxLayout()

        # 中间左边选择框布局
        left_layout = QVBoxLayout()
        # 设置提示标签
        middle_left_label = QLabel("输入图像类型：")
        self.button_group = QButtonGroup(self)
        self.fret_radio = QRadioButton("FRET 三通道")
        self.mitochondria_radio = QRadioButton("线粒体")
        self.nucleus_radio = QRadioButton("细胞核")
        self.cytoplasm_radio = QRadioButton("细胞质")
        self.nucleus_foxo3a_radio = QRadioButton("细胞核 - foxo3a")
        self.nucleus_mitochondria_radio = QRadioButton("细胞核 - 线粒体(优先)")
        self.nucleus_mitochondria_radio.setChecked(True)
        self.nucleus_cytoplasm_radio = QRadioButton("细胞核 - 细胞质(优先)")
        # TODO 没有开放该按钮
        self.cytoplasm_radio.setCheckable(False)
        self.nucleus_cytoplasm_radio.setCheckable(False)

        self.button_group.addButton(self.fret_radio)
        self.button_group.addButton(self.mitochondria_radio)
        self.button_group.addButton(self.nucleus_radio)
        self.button_group.addButton(self.cytoplasm_radio)
        self.button_group.addButton(self.nucleus_foxo3a_radio)
        self.button_group.addButton(self.nucleus_mitochondria_radio)
        self.button_group.addButton(self.nucleus_cytoplasm_radio)

        left_layout.addWidget(middle_left_label)
        left_layout.addWidget(self.fret_radio)
        left_layout.addWidget(self.mitochondria_radio)
        left_layout.addWidget(self.nucleus_radio)
        left_layout.addWidget(self.cytoplasm_radio)
        left_layout.addWidget(self.nucleus_foxo3a_radio)
        left_layout.addWidget(self.nucleus_mitochondria_radio)
        left_layout.addWidget(self.nucleus_cytoplasm_radio)
        middle_layout.addLayout(left_layout, 1)

        # 中间右边参数输入框布局
        right_layout = QVBoxLayout()
        middle_left_label = QLabel("输入细胞直径参数(像素点为单位)：")
        right_layout.addWidget(middle_left_label)

        cell_diameter_label = QLabel("细胞直径:    ")
        self.cell_diameter_input = QLineEdit()
        self.cell_diameter_input.setText("200")
        cell_diameter_layout = QHBoxLayout()
        cell_diameter_layout.addWidget(cell_diameter_label)
        cell_diameter_layout.addWidget(self.cell_diameter_input)
        right_layout.addLayout(cell_diameter_layout)

        cell_min_diameter_label = QLabel("细胞最小直径:")
        self.cell_min_diameter_input = QLineEdit()
        self.cell_min_diameter_input.setText("100")
        cell_min_diameter_layout = QHBoxLayout()
        cell_min_diameter_layout.addWidget(cell_min_diameter_label)
        cell_min_diameter_layout.addWidget(self.cell_min_diameter_input)
        right_layout.addLayout(cell_min_diameter_layout)

        cell_max_diameter_label = QLabel("细胞最大直径:")
        self.cell_max_diameter_input = QLineEdit()
        self.cell_max_diameter_input.setText("500")
        cell_max_diameter_layout = QHBoxLayout()
        cell_max_diameter_layout.addWidget(cell_max_diameter_label)
        cell_max_diameter_layout.addWidget(self.cell_max_diameter_input)
        right_layout.addLayout(cell_max_diameter_layout)

        nuclei_diameter_label = QLabel("核直径:      ")
        self.nuclei_diameter_input = QLineEdit()
        self.nuclei_diameter_input.setText("120")
        nuclei_diameter_layout = QHBoxLayout()
        nuclei_diameter_layout.addWidget(nuclei_diameter_label)
        nuclei_diameter_layout.addWidget(self.nuclei_diameter_input)
        right_layout.addLayout(nuclei_diameter_layout)

        nuclei_min_diameter_label = QLabel("核最小直径:  ")
        self.nuclei_min_diameter_input = QLineEdit()
        self.nuclei_min_diameter_input.setText("80")
        nuclei_min_diameter_layout = QHBoxLayout()
        nuclei_min_diameter_layout.addWidget(nuclei_min_diameter_label)
        nuclei_min_diameter_layout.addWidget(self.nuclei_min_diameter_input)
        right_layout.addLayout(nuclei_min_diameter_layout)

        nuclei_max_diameter_label = QLabel("核最大直径:  ")
        self.nuclei_max_diameter_input = QLineEdit()
        self.nuclei_max_diameter_input.setText("200")
        nuclei_max_diameter_layout = QHBoxLayout()
        nuclei_max_diameter_layout.addWidget(nuclei_max_diameter_label)
        nuclei_max_diameter_layout.addWidget(self.nuclei_max_diameter_input)
        right_layout.addLayout(nuclei_max_diameter_layout)

        middle_layout.addLayout(right_layout, 1)

        main_layout.addLayout(middle_layout)

        cmd_label = QLabel("运行输出日志 :")
        cmd_label.setFont(font)
        main_layout.addWidget(cmd_label)

        # 底部运行输出文本框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setLineWrapMode(QTextEdit.NoWrap)  # 禁止自动换行
        main_layout.addWidget(self.output_text)

        # 运行和终止按钮布局
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("运行")
        self.run_button.setStyleSheet("background-color: green; color: white;")
        self.run_button.clicked.connect(self.run)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("终止")
        self.stop_button.setStyleSheet("background-color: red; color: white;")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if folder:
            self.input_folder_input.setText(folder)

    def run(self):
        if self.running:
            return
        self.running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_event.clear()  # 清除停止事件
        input_folder = self.input_folder_input.text().strip()
        self.output_text.clear()
        # 重定向标准输出
        original_stdout = sys.stdout
        output_redirector = OutputRedirector(self.output_text)
        sys.stdout = output_redirector

        self.output_text.append("输入文件路径为: " + str(input_folder))
        self.output_text.append("开始运行==========================================>图像单细胞分割程序")

        try:
            # 为运行操作创建一个新线程
            threading.Thread(target=self._run_segmentation, args=(input_folder, output_redirector)).start()
        except Exception as e:
            self.output_text.append("运行出错==============================================>" + str(e))
            self.running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout

    def _run_segmentation(self, input_folder, output_redirector):
        original_stdout = sys.stdout
        sys.stdout = output_redirector
        try:
            # 获取当前被选中的按钮
            checked_button = self.button_group.checkedButton()
            if checked_button == self.fret_radio:
                # 执行 FRET 三通道的分割操作
                print("执行 FRET 三通道的分割操作")
            elif checked_button == self.mitochondria_radio:
                # 执行线粒体的分割操作
                print("执行线粒体的分割操作")
            elif checked_button == self.nucleus_radio:
                # 执行细胞核的分割操作
                print("执行细胞核的分割操作")
            elif checked_button == self.nucleus_foxo3a_radio:
                # 执行细胞核 - foxo3a分割操作
                print("执行细胞核 - foxo3a分割操作")
                from segmentation.nuclei_foxo3a_seg import FOXO3ANucleiSegmentation
                model = FOXO3ANucleiSegmentation(
                    seg_diameter=int(self.cell_diameter_input.text()),
                    seg_min_diameter=int(self.cell_min_diameter_input.text()),
                    seg_max_diameter=int(self.cell_max_diameter_input.text()),
                    seg_nuclei_diameter=int(self.nuclei_diameter_input.text()),
                    seg_nuclei_min_diameter=int(self.nuclei_min_diameter_input.text()),
                    seg_nuclei_max_diameter=int(self.nuclei_max_diameter_input.text()),
                    output_redirector=output_redirector
                )

                def start_model(image_set_path, seg_model):
                    if self.stop_event.is_set():
                        return
                    return seg_model.start(image_set_path)

                batch = BatchProcessing(input_folder, stop_event=self.stop_event, csv_name='foxo3a.csv')
                batch.start(start_model, model)  # 传递 stop_event

            elif checked_button == self.nucleus_mitochondria_radio:
                # 执行细胞核 - 线粒体的分割操作
                print("执行细胞核 - 线粒体的分割操作")
                # 执行细胞核 - 线粒体的分割操作
                from segmentation.nuclei_mit_seg import MitNucleiSegmentation

                model = MitNucleiSegmentation(
                    seg_diameter=int(self.cell_diameter_input.text()),
                    seg_min_diameter=int(self.cell_min_diameter_input.text()),
                    seg_max_diameter=int(self.cell_max_diameter_input.text()),
                    seg_nuclei_diameter=int(self.nuclei_diameter_input.text()),
                    seg_nuclei_min_diameter=int(self.nuclei_min_diameter_input.text()),
                    seg_nuclei_max_diameter=int(self.nuclei_max_diameter_input.text()),
                    output_redirector=output_redirector
                )

                def start_model(image_set_path, seg_model):
                    if self.stop_event.is_set():
                        return
                    seg_model.start(image_set_path)

                batch = BatchProcessing(input_folder, stop_event=self.stop_event)
                batch.start(start_model, model)  # 传递 stop_event

            elif checked_button == self.nucleus_cytoplasm_radio:
                # 执行细胞核 - 细胞质的分割操作
                print("执行细胞核 - 细胞质的分割操作")
            print("完成分割操作！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        except Exception as e:
            print("运行出错==============================================>" + str(e))
        finally:
            self.running = False
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            sys.stdout = original_stdout

    def stop(self):
        self.running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_event.set()  # 设置停止事件
        self.output_text.append("运行已终止==============================================>")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SegmentationUI()
    window.show()
    sys.exit(app.exec_())
