# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'playvideo.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import glob
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(713, 633)
        self.folder = None
        self.imgnum = None
        self.imagelist = None
        self.playstate = True
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.oneframeforward = QtWidgets.QPushButton(self.centralwidget)
        self.oneframeforward.setObjectName("oneframeforward")
        self.oneframeforward.clicked.connect(self.jumponeframeforward)
        self.gridLayout.addWidget(self.oneframeforward, 1, 4, 1, 1)
        self.slidervideo = QtWidgets.QSlider(self.centralwidget)
        self.slidervideo.setOrientation(QtCore.Qt.Horizontal)
        self.slidervideo.setObjectName("sildervideo")
        self.slidervideo.valueChanged.connect(self.setframe)
        self.gridLayout.addWidget(self.slidervideo, 1, 2, 1, 1)
        self.stopvideo = QtWidgets.QPushButton(self.centralwidget)
        self.stopvideo.setObjectName("stopvideo")
        self.stopvideo.clicked.connect(self.stopclick)
        self.gridLayout.addWidget(self.stopvideo, 1, 1, 1, 1)
        self.oneframeback = QtWidgets.QPushButton(self.centralwidget)
        self.oneframeback.setObjectName("oneframeback")
        self.oneframeback.clicked.connect(self.jumponeframeback)
        self.gridLayout.addWidget(self.oneframeback, 1, 3, 1, 1)
        self.startvideo = QtWidgets.QPushButton(self.centralwidget)
        self.startvideo.setAutoFillBackground(False)
        self.startvideo.setAutoDefault(False)
        self.startvideo.setDefault(False)
        self.startvideo.setFlat(False)
        self.startvideo.setObjectName("startvideo")
        self.gridLayout.addWidget(self.startvideo, 1, 0, 1, 1)
        self.startvideo.clicked.connect(self.startclick)
        self.Image = QtWidgets.QGraphicsView(self.centralwidget)
        self.Image.setObjectName("Image")
        self.gridLayout.addWidget(self.Image, 0, 0, 1, 5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 713, 19))
        self.menubar.setObjectName("menubar")
        self.menuVideo_tool = QtWidgets.QMenu(self.menubar)
        self.menuVideo_tool.setObjectName("menuVideo_tool")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSelect_file = QtWidgets.QAction(MainWindow)
        self.actionSelect_file.setObjectName("actionSelect_file")
        self.actionSelect_file.triggered.connect(self.selectfile)
        self.actionSelect_folder = QtWidgets.QAction(MainWindow)
        self.actionSelect_folder.setObjectName("actionSelect_folder")
        self.actionSelect_folder.triggered.connect(self.selectfolder)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuVideo_tool.addAction(self.actionSelect_file)
        self.menuVideo_tool.addAction(self.actionSelect_folder)
        self.menuVideo_tool.addAction(self.actionExit)
        self.menubar.addAction(self.menuVideo_tool.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.oneframeforward.setText(_translate("MainWindow", ">"))
        self.stopvideo.setText(_translate("MainWindow", "Stop"))
        self.oneframeback.setText(_translate("MainWindow", "<"))
        self.startvideo.setText(_translate("MainWindow", "Start"))
        self.menuVideo_tool.setTitle(_translate("MainWindow", "File"))
        self.actionSelect_file.setText(_translate("MainWindow", "Select file"))
        self.actionSelect_folder.setText(_translate("MainWindow", "Select folder"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def startclick(self):
        if self.folder:
            self.playstate = True
            self.scene = QtWidgets.QGraphicsScene()
            start = self.imgnum
            l = len(self.imagelist)
            for image in self.imagelist[start:]:
                if self.playstate == True:
                    self.scene.addPixmap(QtGui.QPixmap(image))
                    self.Image.setScene(self.scene)
                    QtGui.QGuiApplication.processEvents()
                    self.imgnum += 1

                    #self.slidervideo.setValue(int(100*self.imgnum/l))
                    #print(int(100*self.imgnum/l), self.slidervideo.value())
                    #print('startclick:',  self.imgnum)
                else:
                    break
        else:
            print('Select a folder or image first')

    def stopclick(self):
        self.playstate = False
        print(self.imgnum, self.playstate)

    def selectfile(self):
        file = QtWidgets.QFileDialog.getOpenFileName()
        if file[0].split('.')[-1] == 'png':
            self.folder = file
            self.scene = QtWidgets.QGraphicsScene()
            self.scene.addPixmap(QtGui.QPixmap(str(file[0])))
            self.Image.setScene(self.scene)
        else:
            print('File must be .png')

    def selectfolder(self):
        #file = QtWidgets.QFileDialog.getExistingDirectory()
        file = '/home/lea/Dokumente/FSU/Anwendungspraktikum/Videos/GoPro/GoProFrames'
        self.folder = file
        self.imgnum = 1
        self.imagelist = sorted(glob.glob(os.path.join(self.folder + '/*.png')))
        file = self.imagelist[0]
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.addPixmap(QtGui.QPixmap(file))
        self.Image.setScene(self.scene)

    def setframe(self):
        if self.folder:
            self.playstate = False
            size = self.slidervideo.value()
            range = len(glob.glob(os.path.join(self.folder + '/*.png')))
            num = int(size / 100 * range)
            self.imgnum = num
            self.scene.addPixmap(QtGui.QPixmap(self.imagelist[self.imgnum]))
            self.Image.setScene(self.scene)

    def jumponeframeback(self):
        self.playstate = False
        print(self.imgnum)
        self.imgnum -= 1
        print('AFter: ', self.imgnum)
        self.scene.addPixmap(QtGui.QPixmap(self.imagelist[self.imgnum]))
        self.Image.setScene(self.scene)

    def jumponeframeforward(self):
        if self.playstate:
            self.playstate = False
        else:
            self.imgnum += 1
            self.scene.addPixmap(QtGui.QPixmap(self.imagelist[self.imgnum]))
            self.Image.setScene(self.scene)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

