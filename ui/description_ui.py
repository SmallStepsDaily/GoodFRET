import os
import sys
import markdown
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
import re

# 假设 current_dir 是当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))


class MarkdownReaderUI(QWidget):
    def __init__(self):
        super().__init__()
        self.web_view = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.web_view = QWebEngineView()
        # 设置允许加载本地资源
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        # 拼接 Markdown 文件的绝对路径
        file_path = os.path.join(current_dir, 'markdown', '命名规范介绍.md')

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                markdown_text = file.read()

            # 将 Markdown 文本转换为 HTML，同时支持表格、脚注和代码块扩展
            html = markdown.markdown(markdown_text, extensions=['tables', 'footnotes', 'fenced_code'])

            # 添加自定义 CSS 样式来增大字体大小、设置代码块样式和图像大小
            css = """
            <style>
                body { font-size: 25px; }
                pre {
                    background-color: #f4f4f4;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow-x: auto;
                }
                code {
                    font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
            """
            html = f'{css}{html}'

            # 在 QWebEngineView 中加载 HTML 内容
            # 在这里要添加本地的安全路径进行识别
            base_url = QUrl.fromLocalFile(file_path)
            self.web_view.setHtml(html, base_url)
        except Exception as e:
            print(f'读取文件时出错: {e}')

        layout.addWidget(self.web_view)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MarkdownReaderUI()
    window.show()
    sys.exit(app.exec_())
