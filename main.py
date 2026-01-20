from gui.gui import PostProcessingApp
from PyQt6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setStyle("Fusion")
    window = PostProcessingApp()
    window.showMaximized()
    sys.exit(app.exec())
