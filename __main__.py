import random
import sys
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QVBoxLayout
import numpy as np
import pandas as pd
import binvox_rw
import vtkplotlib as vpl
from stl.mesh import Mesh
import tkinter as tk
from tkinter import filedialog
import subprocess
from shutil import copyfile
import os
import numpy as np
import tensorflow
import keras
from keras import regularizers
import os
import random
from keras.layers import Input, MaxPooling3D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv3D
from keras.models import Model
from keras import optimizers
from keras import regularizers

copied_stl_path = ".\\out\\input.stl"


class MyWidget(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.items = 0
        self.select_stl_btn = QtWidgets.QPushButton("Select Stl Model")
        self.view_stl_btn = QtWidgets.QPushButton("View Stl Model")
        self.convert_stl_to_binvox_btn = QtWidgets.QPushButton("Convert to Binvox")
        self.view_binvox_btn = QtWidgets.QPushButton("View Binvox Model")
        self.predict_btn = QtWidgets.QPushButton("Predict model")

        # default disabled buttons
        self.view_stl_btn.setEnabled(False)
        self.convert_stl_to_binvox_btn.setEnabled(False)
        self.view_binvox_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)

        self.select_stl_btn.clicked.connect(self.select_stl)
        self.view_stl_btn.clicked.connect(self.show_stl)
        self.convert_stl_to_binvox_btn.clicked.connect(self.convert_binvox)
        self.view_binvox_btn.clicked.connect(self.show_binvox)
        self.predict_btn.clicked.connect(self.predict_out)

        layout = QVBoxLayout()

        layout.addWidget(self.select_stl_btn)
        layout.addWidget(self.view_stl_btn)
        layout.addWidget(self.convert_stl_to_binvox_btn)
        layout.addWidget(self.view_binvox_btn)
        layout.addWidget(self.predict_btn)

        # Set the layout on the application's window
        self.setLayout(layout)

    @QtCore.Slot()
    def select_stl(self):
        global stl_path
        root = tk.Tk()
        root.withdraw()

        stl_path = filedialog.askopenfilename()

        # enable next step buttons
        if stl_path.endswith('.stl'):
            self.view_stl_btn.setEnabled(True)
            self.convert_stl_to_binvox_btn.setEnabled(True)

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
        global copied_stl_path
        global stl_path

        # clean create output directory
        subprocess.call('rmdir /q /s out', shell=True)
        subprocess.call('mkdir out', shell=True)
        copyfile(stl_path, copied_stl_path)
        subprocess.call(['.\\executables\\binvox.exe', '-c', '-d', '64', copied_stl_path])

        # enable next step buttons
        self.view_binvox_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)

    # binvox model görüntüleme
    @QtCore.Slot()
    def show_binvox(self):
        with open("./out/input.binvox", 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        from mayavi import mlab

        xx, yy, zz = np.where(model.data == 1)

        mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(0, 1, 0),
                             scale_factor=1)

        mlab.show()

    # predict
    @QtCore.Slot()
    def predict_out(self):
        
        #predict of real parts with model 2
        model = keras.models.Sequential()
        model.add(Conv3D(32, 7, strides=2, padding='valid', activation='relu', input_shape = (64,64,64,1)))
        model.add(Conv3D(32, 5, strides=1, padding='same', activation='relu'))
        model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
        model.add(Conv3D(64, 3, strides=1, padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
        model.add(Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.0)))
        model.add(Dense(2, activation = 'sigmoid'))
        model.load_weights('models/model_mpi.h5')
        
        batch_input = []
        with open("./out/input.binvox", 'rb') as file:
            try:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
            except:
                print(file)
        model_input = np.reshape(data, (64, 64, 64, 1))
        batch_input += [model_input]
        batch_x = np.array(batch_input)
        
        result = model.predict(batch_x)
        result[:,0][result[:,0] >= 0.5] = 1
        result[:,1][result[:,1] >= 0.5] = 1
        result[:,1][result[:,1] < 0.5] = 0
        result[:,0][result[:,0] < 0.5] = 0
        print("Classification Results", result)

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
    window.setWindowTitle("MPI CS491/2")
    window.show()

    sys.exit(app.exec())
