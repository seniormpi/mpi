import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
import stltovoxel
import binvox_rw
import numpy as np

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World",
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))
        print("----")
        stltovoxel.convert_file('3DBenchy.stl', 'output.npy')
        print("stl to voxel finished")
        data = np.load('output.npy')
        print(data)
        voxel = binvox_rw.Voxels(data, )
        with open("o.binwox", "w") as fp:
            binvox_rw.write(voxel, fp)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())