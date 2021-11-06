import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
import stltovoxel
import binvox_rw
import numpy as np
import vtkplotlib as vpl
from stl.mesh import Mesh


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.items = 0

        self.material = QtWidgets.QLineEdit()
        self.stlm = QtWidgets.QPushButton("View Stl Model")
        self.tobinvox = QtWidgets.QPushButton("Convert to Binvox")
        self.binvoxm = QtWidgets.QPushButton("View Binvox Model")
        self.predict = QtWidgets.QPushButton("Predict model")

        #right
        self.right = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.TopToBottom)
        self.right.addWidget(QtWidgets.QLabel("Material: "))
        self.right.addWidget(self.material)
        self.right.addWidget(self.stlm)
        self.right.addWidget(self.tobinvox)
        self.right.addWidget(self.binvoxm)
        self.right.addWidget(self.predict)

        self.layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.Up)

        self.layout.addLayout(self.right)

        self.setLayout(self.layout)

        self.stlm.clicked.connect(self.show_stl)
        self.tobinvox.clicked.connect(self.convert_binvox)
        self.binvoxm.clicked.connect(self.show_binvox)
        self.predict.clicked.connect(self.predict_out)

    #stl model gösterme
    @QtCore.Slot()
    def show_stl(self):

        mesh = Mesh.from_file("Mill2.stl")
        fig = vpl.figure()
        mesh = vpl.mesh_plot(mesh)
        vpl.show()

    #stl to binvox convert
    @QtCore.Slot()
    def convert_binvox(self):
    
        stltovoxel.convert_file('3DBenchy.stl', 'output.npy')
        print("stl to voxel finished")
        data = np.int32(np.load('output.npy'))
        print(data)

        with open("3dbenchy.binvox", 'rb') as file:
            data2 = np.int32(binvox_rw.read_as_3d_array(file).data)
            print(data2)

    #binvox model görüntüleme
    @QtCore.Slot()
    def show_binvox(self):

        print("will show binvox")
        #TODO binvox gösterme

    #predict
    @QtCore.Slot()
    def predict_out(self):
        print("will predict")
    
        
    
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, widget):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Machining Process Aid")

        #menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        #exit Qaction
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
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec())