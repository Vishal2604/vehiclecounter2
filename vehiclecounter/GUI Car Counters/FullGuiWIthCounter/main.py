import sys
from PySide6.QtWidgets import QApplication
from single_instance import SingleInstance
from gui import MainWindow 
import cv2git
if __name__ == "__main__":
    instance_checker = SingleInstance()
    instance_checker.acquire()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print(cv2.getBuildInformation())
    sys.exit(app.exec())