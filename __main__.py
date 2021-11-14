import sys
from PySide6 import QtCore, QtWidgets, QtGui
import binvox_rw
import numpy as np
import vtkplotlib as vpl
from stl.mesh import Mesh
import tkinter as tk
from tkinter import filedialog
import subprocess
import os


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.items = 0
        self.select_stl_btn = QtWidgets.QPushButton("Select Stl Model")
        self.view_stl_btn = QtWidgets.QPushButton("View Stl Model")
        self.convert_stl_to_binwox_btn = QtWidgets.QPushButton("Convert to Binvox")
        self.view_binwox_btn = QtWidgets.QPushButton("View Binvox Model")
        self.predict_btn = QtWidgets.QPushButton("Predict model")

        self.view_stl_btn.setEnabled(False)
        self.convert_stl_to_binwox_btn.setEnabled(False)
        self.view_binwox_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)

        # right
        self.right = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.TopToBottom)
        self.right.addWidget(self.select_stl_btn)
        self.right.addWidget(self.view_stl_btn)
        self.right.addWidget(self.convert_stl_to_binwox_btn)
        self.right.addWidget(self.view_binwox_btn)
        self.right.addWidget(self.predict_btn)

        self.layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.Up)

        self.layout.addLayout(self.right)
        self.setLayout(self.layout)

        self.select_stl_btn.clicked.connect(self.select_stl)
        self.view_stl_btn.clicked.connect(self.show_stl)
        self.convert_stl_to_binwox_btn.clicked.connect(self.convert_binvox)
        self.view_binwox_btn.clicked.connect(self.show_binvox)
        self.predict_btn.clicked.connect(self.predict_out)

    @QtCore.Slot()
    def select_stl(self):
        global stl_path
        root = tk.Tk()
        root.withdraw()

        stl_path = filedialog.askopenfilename()

        self.view_stl_btn.setEnabled(True)
        self.convert_stl_to_binwox_btn.setEnabled(True)

    # stl model gösterme
    @QtCore.Slot()
    def show_stl(self):
        mesh = Mesh.from_file(stl_path)
        fig = vpl.figure()
        mesh = vpl.mesh_plot(mesh)
        vpl.show()

    # stl to binvox convert
    @QtCore.Slot()
    def convert_binvox(self):
        subprocess.call(['binvox.exe', '-c', '-d', '100', stl_path])

        self.view_binwox_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)

    # binvox model görüntüleme
    @QtCore.Slot()
    def show_binvox(self):
        print(os.path.splitext(os.path.basename(stl_path))[0].lower())
        with open(os.path.splitext(os.path.basename(stl_path).lower())[0] + ".binvox", 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.voxels(model.data, edgecolor="k")
        plt.show()

    # predict
    @QtCore.Slot()
    def predict_out(self):
        print("will predict")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, widget):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Machining Process Aid")

        # menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        # exit Qaction
        exit_action = QtGui.QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @QtCore.Slot()
    def exit_app(self, checked):
        QtWidgets.QApplication.quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    widget = MyWidget()
    window = MainWindow(widget)
    window.resize(600, 400)
    window.show()

    sys.exit(app.exec())
