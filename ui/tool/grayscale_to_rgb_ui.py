import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class GrayscaleToRGBUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.converted_image = None

    def initUI(self):
        self.setFixedSize(1280, 720)
        layout = QHBoxLayout()
        # 设置窗口图标
        self.setWindowIcon(QIcon('logo.jpg'))  # 加载图标文件
        # 左边图像输入部分
        self.image_label = ClickableLabel('输入图像', self)
        self.image_label.setFixedSize(700, 700)
        self.image_label.setStyleSheet('border: 2px dashed gray;')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setAcceptDrops(True)
        self.image_label.clicked.connect(self.open_image)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)

        # 右边参数输入和按钮部分
        # RGB 参数输入
        r_label = QLabel('R (0 - 255):', self)
        self.r_edit = QLineEdit(self)
        self.r_edit.setPlaceholderText('输入 R 值')

        g_label = QLabel('G (0 - 255):', self)
        self.g_edit = QLineEdit(self)
        self.g_edit.setPlaceholderText('输入 G 值')

        b_label = QLabel('B (0 - 255):', self)
        self.b_edit = QLineEdit(self)
        self.b_edit.setPlaceholderText('输入 B 值')

        # 转换图像高度和宽度输入
        height_label = QLabel('转换图像高度 (像素):', self)
        self.height_edit = QLineEdit(self)
        self.height_edit.setPlaceholderText('输入高度值')
        self.height_edit.setText('512')

        width_label = QLabel('转换图像宽度 (像素):', self)
        self.width_edit = QLineEdit(self)
        self.width_edit.setPlaceholderText('输入宽度值')
        self.width_edit.setText('512')

        self.convert_button = QPushButton('转换', self)
        self.convert_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.convert_button.clicked.connect(self.convert_image)

        self.save_button = QPushButton('保存', self)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        self.save_button.clicked.connect(self.save_image)

        right_layout = QVBoxLayout()
        right_layout.addWidget(r_label)
        right_layout.addWidget(self.r_edit)
        right_layout.addWidget(g_label)
        right_layout.addWidget(self.g_edit)
        right_layout.addWidget(b_label)
        right_layout.addWidget(self.b_edit)
        right_layout.addWidget(height_label)
        right_layout.addWidget(self.height_edit)
        right_layout.addWidget(width_label)
        right_layout.addWidget(self.width_edit)
        right_layout.addWidget(self.convert_button)
        right_layout.addWidget(self.save_button)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        self.setLayout(layout)
        self.setWindowTitle('图像灰度转 RGB 工具')

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            self.image_path = url.toLocalFile()
            if self.image_path.lower().endswith('.tiff') or self.image_path.lower().endswith('.tif'):
                # 读取并压缩图像
                img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (512, 512))
                height, width = img.shape
                qImg = QImage(img.data, width, height, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qImg)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
                event.acceptProposedAction()
            else:
                print('请拖入 TIFF 格式的图像')

    @pyqtSlot()
    def open_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, '打开图像', '', 'TIFF 文件 (*.tiff *.tif)')
        if self.image_path:
            # 读取并压缩图像
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512))
            height, width = img.shape
            qImg = QImage(img.data, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    @pyqtSlot()
    def convert_image(self):
        if self.image_path:
            try:
                # 禁用输入图像功能
                self.image_label.setEnabled(False)
                r = int(self.r_edit.text())
                g = int(self.g_edit.text())
                b = int(self.b_edit.text())
                height = int(self.height_edit.text())
                width = int(self.width_edit.text())

                if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                    print('请输入有效的 RGB 值（0 - 255）')
                    return

                gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                gray_image = cv2.resize(gray_image, (512, 512))
                colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
                for y in range(gray_image.shape[0]):
                    for x in range(gray_image.shape[1]):
                        gray_value = gray_image[y, x]
                        r_value = int(gray_value * (r / 255))
                        g_value = int(gray_value * (g / 255))
                        b_value = int(gray_value * (b / 255))
                        colored_image[y, x] = [b_value, g_value, r_value]

                self.converted_image = cv2.resize(colored_image, (width, height))
                # 将转换后的图像显示在左侧预览框
                height, width, channel = self.converted_image.shape
                bytesPerLine = 3 * width
                qImg = QImage(self.converted_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
                print('图像转换完成')
            except ValueError:
                print('请输入有效的数值')
            except Exception as e:
                print(f'转换过程中出现错误: {e}')
            finally:
                # 恢复输入图像功能
                self.image_label.setEnabled(True)
        else:
            print('请先拖入或打开图像')

    @pyqtSlot()
    def save_image(self):
        if self.converted_image is not None:
            file_dialog = QFileDialog()
            save_path, _ = file_dialog.getSaveFileName(self, '保存图像', '', '图像文件 (*.png *.jpg)')
            if save_path:
                try:
                    cv2.imwrite(save_path, self.converted_image)
                    print(f'图像已保存到: {save_path}')
                except Exception as e:
                    print(f'保存图像时出现错误: {e}')
        else:
            print('请先进行图像转换')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GrayscaleToRGBUI()
    window.show()
    sys.exit(app.exec_())
