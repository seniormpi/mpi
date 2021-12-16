import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
import numpy as np
import binvox_rw
import vtkplotlib as vpl
from stl.mesh import Mesh
import tkinter as tk
from tkinter import filedialog
import subprocess
from shutil import copyfile
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
    SceneEditor
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item

from predict import predict

copied_stl_path = ".\\out\\input.stl"

# The actual visualization
class VisualizationBinvox(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some

        # We can do normal mlab calls on the embedded scene.
        with open("./out/input.binvox", 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        xx, yy, zz = np.where(model.data == 1)
        self.scene.mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)
        #self.scene.mlab.test_points3d()

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )

################################################################################
class BinvoxWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = VisualizationBinvox()

        # If you want to debug, beware that you need to remove the Qt input hook.
        # QtCore.pyqtRemoveInputHook()
        # import pdb ; pdb.set_trace()
        # QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.

        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


class MyWidget(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.items = 0
        self.select_stl_btn = QtWidgets.QPushButton("Select Stl Model")
        self.convert_stl_to_binvox_btn = QtWidgets.QPushButton("Convert to Binvox")
        self.predict_btn = QtWidgets.QPushButton("Predict model")

        # default disabled buttons
        self.convert_stl_to_binvox_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)

        self.select_stl_btn.clicked.connect(self.select_stl)
        self.convert_stl_to_binvox_btn.clicked.connect(self.convert_binvox)
        self.predict_btn.clicked.connect(self.predict_out)

        horizontal_layout_1 = QHBoxLayout()

        vertical_layout_1 = QVBoxLayout()
        vertical_layout_1.addWidget(self.select_stl_btn)
        vertical_layout_1.addWidget(self.convert_stl_to_binvox_btn)
        vertical_layout_1.addWidget(self.predict_btn)

        self.stl_label = QLabel("Please select a STL file", self)
        self.res_label = QLabel("Results: ", self)
        self.result_label = QLabel(" ", self)
        self.result_label2 = QLabel(" ", self)




        self.vertical_layout_3 = QVBoxLayout()

        self.vertical_layout_3.addWidget(self.stl_label, alignment=QtCore.Qt.AlignTop)
        vertical_layout_1.addWidget(QLabel("", self), alignment=QtCore.Qt.AlignCenter)
        vertical_layout_1.addWidget(QLabel("", self), alignment=QtCore.Qt.AlignCenter)
        vertical_layout_1.addWidget(self.res_label, alignment=QtCore.Qt.AlignCenter)
        vertical_layout_1.addWidget(self.result_label, alignment=QtCore.Qt.AlignCenter)
        vertical_layout_1.addWidget(self.result_label2, alignment=QtCore.Qt.AlignCenter)
        

        self.stl_widget = vpl.QtFigure()
        self.vertical_layout_3.addWidget(self.stl_widget)



        horizontal_layout_1.addLayout(vertical_layout_1)
        horizontal_layout_1.addLayout(vertical_layout_1)
        horizontal_layout_1.addLayout(self.vertical_layout_3)

        # Set the layout on the application's window
        self.setLayout(horizontal_layout_1)

    def select_stl(self):
        global stl_path
        root = tk.Tk()
        root.withdraw()

        stl_path = filedialog.askopenfilename()

        # enable next step buttons
        if stl_path.endswith('.stl'):
            self.convert_stl_to_binvox_btn.setEnabled(True)
            self.stl_label.setText(stl_path)
            self.show_stl()

    # stl model gösterme
    def show_stl(self):
        mesh = Mesh.from_file(stl_path)
        vpl.mesh_plot(mesh, fig=self.stl_widget)
        self.stl_widget.show()

    # stl to binvox convert
    def convert_binvox(self):
        global copied_stl_path
        global stl_path

        # clean create output directory
        subprocess.call('rmdir /q /s out', shell=True)
        subprocess.call('mkdir out', shell=True)
        copyfile(stl_path, copied_stl_path)
        subprocess.call(['.\\executables\\binvox.exe', '-c', '-d', '64', copied_stl_path])

        # enable next step buttons
        self.predict_btn.setEnabled(True)
        self.show_binvox()

    # binvox model görüntüleme
    def show_binvox(self):
        container = QtWidgets.QWidget()
        self.vertical_layout_3.addWidget(BinvoxWidget(container))

    # predict
    def predict_out(self):
        print()
        a = "The procedure: "
        a += predict().predict_mpi()
        self.result_label2.setText(a)
        b = "The Machinability: "
        b += predict().predict_mach()
        self.result_label.setText(b)
        print(b)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, widget):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Machining Process Aid")

        # menu
        self.menu = self.menuBar()
        self.setCentralWidget(widget)

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
