from PyQt5.QtWidgets import QApplication
from gui import EmotionApp

if __name__ == "__main__":
    app = QApplication([])
    window = EmotionApp()
    window.show()
    app.exec_()
