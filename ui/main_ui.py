import sys
import platform

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QGuiApplication
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QTabWidget
)

# 在创建 QApplication 之前设置属性
QtCore.QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

# 检查操作系统类型
is_macos = platform.system() == 'Darwin'


class ImageProcessingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tab_widget = None
        self.initUI()

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('GoodFRET')

        # 根据不同系统设置不同的初始大小
        if is_macos:
            self.resize(1600, 900)  # macOS上使用更合适的初始大小
        else:
            self.setGeometry(100, 100, 1920, 1080)

        self.center()

        # 设置窗口图标
        try:
            self.setWindowIcon(QIcon('logo.jpg'))  # 加载图标文件
        except Exception as e:
            print(f"无法加载图标: {e}")

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建中部操作区的标签页
        actions = ['数据说明', '名称修改', '图像分割', 'FRET特征', '表型特征', 'FRET分析', '表型分析', '药效分析',
                   '帮助']
        self.tab_widget = QTabWidget()

        # 在macOS上使用更适合的样式
        if is_macos:
            self.tab_widget.setStyleSheet("""
                QTabBar::tab {
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QTabWidget::pane {
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 10px;
                }
            """)

        for i in range(len(actions)):
            tab = QWidget()
            # 这里可以根据不同的标签页加载不同的界面
            if actions[i] == '名称修改':
                from ui.rename_ui import RenameUI
                layout = QVBoxLayout()
                rename_ui = RenameUI()
                layout.addWidget(rename_ui)
                tab.setLayout(layout)
            elif actions[i] == '图像分割':
                from ui.segmentation_ui import SegmentationUI
                layout = QVBoxLayout()
                segmentation_ui = SegmentationUI()
                layout.addWidget(segmentation_ui)
                tab.setLayout(layout)
            elif actions[i] == '数据说明':
                from ui.description_ui import MarkdownReaderUI
                layout = QVBoxLayout()
                description_ui = MarkdownReaderUI()
                layout.addWidget(description_ui)
                tab.setLayout(layout)
            elif actions[i] == 'FRET特征':
                from ui.fret_extraction_ui import FRETExtractionUI
                layout = QVBoxLayout()
                fret_extraction_ui = FRETExtractionUI()
                layout.addWidget(fret_extraction_ui)
                tab.setLayout(layout)
            elif actions[i] == '表型特征':
                from ui.phenotype_extraction_ui import PhenotypeExtractionUI
                layout = QVBoxLayout()
                phenotype_extraction_ui = PhenotypeExtractionUI()
                layout.addWidget(phenotype_extraction_ui)
                tab.setLayout(layout)
            elif actions[i] == 'FRET分析':
                from ui.fret_analysis_ui import FRETAnalysisUI
                layout = QVBoxLayout()
                fret_analysis_ui = FRETAnalysisUI()
                layout.addWidget(fret_analysis_ui)
                tab.setLayout(layout)
            elif actions[i] == '表型分析':
                from ui.phenotype_analysis_ui import PhenotypeAnalysisUI
                layout = QVBoxLayout()
                phenotype_analysis_ui = PhenotypeAnalysisUI()
                layout.addWidget(phenotype_analysis_ui)
                tab.setLayout(layout)
            elif actions[i] == '药效分析':
                from ui.drug_efficacy import DrugEfficacyAnalysisUI
                layout = QVBoxLayout()
                drug_efficacy_analysis_ui = DrugEfficacyAnalysisUI()
                layout.addWidget(drug_efficacy_analysis_ui)
                tab.setLayout(layout)
            elif actions[i] == "帮助":
                from ui.help_ui import HelpUI
                layout = QVBoxLayout()
                help_ui = HelpUI()
                layout.addWidget(help_ui)
                tab.setLayout(layout)

            # 在macOS上设置标签页的字体
            if is_macos:
                font = QFont("Helvetica", 14)
                tab.setFont(font)

            self.tab_widget.addTab(tab, actions[i])

        # 布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        central_widget.setLayout(main_layout)

    def center(self):
        # 获取屏幕尺寸
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        # 获取窗口尺寸
        window_geometry = self.geometry()
        # 计算窗口左上角坐标
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        # 移动窗口到计算得到的位置
        self.move(x, y)


def load_window():
    app = QApplication(sys.argv)

    # 在macOS上使用系统默认字体
    if is_macos:
        font = QFont()  # 使用系统默认字体
        font.setPointSize(14)  # 设置适当的字体大小
    else:
        font = QFont("Arial", 15)

    app.setFont(font)

    # 在macOS上设置高DPI支持
    if is_macos:
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = ImageProcessingUI()
    window.show()
    sys.exit(app.exec_())