import sys
from PyQt5.QtWidgets import QApplication
from gui import EmotionApp
from PyQt5.QtGui import QIcon

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo.ico'))
    ex = EmotionApp()
    ex.show()
    sys.exit(app.exec_())
