import sys
import io
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QBuffer, pyqtSlot
import NeuralNetwork as nn
import utilities as util


class Canvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(560, 560)
        self.setPixmap(pixmap)
        self.last_x, self.last_y = None, None
        self.penColour = QtGui.QColor('#000000')
        self.backgroundColour = QtGui.QColor('#FFFFFF')
        self.resetBackground()

    def mousePressEvent(self, e):
        self.mouseMoveEvent(e)

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()

        painter = QtGui.QPainter(self.pixmap())
        painter.setPen(self.penColour)
        painter.setBrush(self.penColour)
        painter.drawEllipse(self.last_x - 30, self.last_y - 30, 60, 60)
        painter.end()

        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def resetBackground(self):
        painter = QtGui.QPainter(self.pixmap())
        painter.setPen(self.penColour)
        painter.setBrush(self.backgroundColour)
        painter.drawRect(0, 0, 559, 559)
        painter.end()
        self.update()

    def convertPixmapToImage(self):
        painter = QtGui.QPainter(self.pixmap())
        painter.setPen(self.backgroundColour)
        painter.drawRect(0, 0, 559, 559)
        painter.end()
        self.update()
        img = self.pixmap()
        img = img.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        img.save(buffer, "PNG")
        pilIm = Image.open(io.BytesIO(buffer.data()))
        return pilIm


class UiMainWindow(object):

    def setupUi(self, MainWindow, nn):

        self.string = ""

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 810)
        MainWindow.setMinimumSize(QtCore.QSize(1080, 810))
        MainWindow.setMaximumSize(QtCore.QSize(1080, 810))
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName(" centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self. centralWidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.canvas = Canvas()
        self.canvas.setMinimumSize(QtCore.QSize(560, 560))
        self.canvas.setMaximumSize(QtCore.QSize(560, 560))
        self.canvas.setObjectName("canvas")
        self.canvas.setLineWidth(1)
        self.canvas.setFrameShadow(QtWidgets.QFrame.Plain)
        self.gridLayout_3.addWidget(self.canvas, 1, 0, 1, 1, QtCore.Qt.AlignCenter)

        self.descriptionBox = QtWidgets.QTextBrowser(self.centralWidget)
        self.descriptionBox.setMaximumSize(QtCore.QSize(490, 560))
        self.descriptionBox.setMinimumSize(QtCore.QSize(490, 560))
        self.descriptionBox.setObjectName("textBrowser_2")
        self.gridLayout_3.addWidget(self.descriptionBox, 1, 1, 1, 1)

        self.runButton = QtWidgets.QPushButton(self.centralWidget)
        self.runButton.setMinimumSize(QtCore.QSize(100, 60))
        self.runButton.setMaximumSize(QtCore.QSize(70, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.runButton.setFont(font)
        self.runButton.setObjectName("runButton")
        self.runButton.clicked.connect(lambda: self.onRunClick(nn))
        self.gridLayout_3.addWidget(self.runButton, 2, 1, 1, 1, QtCore.Qt.AlignRight)

        self.clearButton = QtWidgets.QPushButton(self.centralWidget)
        self.clearButton.setMinimumSize(QtCore.QSize(100, 60))
        self.clearButton.setMaximumSize(QtCore.QSize(70, 50))
        self.clearButton.setFont(font)
        self.clearButton.setObjectName("clearButton")
        self.clearButton.clicked.connect(lambda: self.onClearClick())
        self.gridLayout_3.addWidget(self.clearButton, 2, 1, 1, 1, QtCore.Qt.AlignLeft)


        self.resetButton = QtWidgets.QPushButton(self.centralWidget)
        self.resetButton.setFont(font)
        self.resetButton.setObjectName("resetButton")
        self.resetButton.setMinimumSize(QtCore.QSize(110, 60))
        self.resetButton.setMaximumSize(QtCore.QSize(110, 60))
        self.resetButton.clicked.connect(lambda: self.onResetClick())
        self.gridLayout_3.addWidget(self.resetButton, 2, 0, 1, 1, QtCore.Qt.AlignLeft)

        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setMinimumSize(QtCore.QSize(1020, 75))
        self.textBrowser.setMaximumSize(QtCore.QSize(1020, 75))
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser.setLineWidth(1)
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout_3.addWidget(self.textBrowser, 0, 0, 1, 2, QtCore.Qt.AlignCenter)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.tabWidget.addTab(self.tab_4, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Neural Network", "Neural Network"))
        self.runButton.setText(_translate("MainWindow", "Run"))
        self.resetButton.setText(_translate("MainWindow", "Reset"))
        self.clearButton.setText(_translate("MainWindow", "Clear"))
        self.textBrowser.setHtml(_translate("MainWindow", "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
                                            "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'MS Shell Dlg 2\'; font-size:22pt;\">Neural Network for Optical Character Recognition</span></p></body></html>"))
        self.terminalUpdate("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Single Digits"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Multiple Digits"))


    def terminalUpdate(self, string):
        _translate = QtCore.QCoreApplication.translate
        html = "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\np, li { white-space: pre-wrap; }\n</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n<p align=\"left\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Terminal\'; color:#00aa00; font-size:16pt;\">Output:"
        string = string.split('\n')
        for i in string:
            if(i != ""):
                html += "</span></p></body></html></style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n<p align=\"left\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Terminal\'; color:#00aa00; font-size:16pt;\">"
                html += i
        html += "</span></p></body></html>"
        self.descriptionBox.setHtml(_translate("MainWindow", html))

    @pyqtSlot()
    def onClearClick(self):
        self.string = ""
        self.terminalUpdate(self.string)
        self.canvas.resetBackground()

    @pyqtSlot()
    def onResetClick(self):
        _translate = QtCore.QCoreApplication.translate
        self.canvas.resetBackground()
        self.string += "Reset!\n------------------------------------------------\n"
        self.terminalUpdate(self.string)

    @pyqtSlot()
    def onRunClick(self, nn):
        finalAnswer = ""
        self.string += "Converting canvas to PIL Image ..."
        self.terminalUpdate(self.string)
        img = self.canvas.convertPixmapToImage()
        self.string += "\nDone!\nConverting PIL Image to greyscale array ..."
        self.terminalUpdate(self.string)
        img = util.loadImageFromPIL(img)
        imgArray = util.cropOutNumbers(img)
        for image in imgArray:
            img = util.centreImage(image)
            data = util.cleanImage(img)
            img = data.reshape((28, 28))
            plt.imshow(img, cmap="Greys")
            plt.show()
            self.string += "\nDone!\nFeeding inputs to neural network ..."
            self.terminalUpdate(self.string)
            answer, activation = nn.getAnswer(data)
            finalAnswer += answer
            self.string += "\nDone!\nAnswer is: " + answer + " with " + activation + " activation.\n------------------------------------------------\n"
        self.string += "\nFinal Answer is: " + finalAnswer + "\n------------------------------------------------\n"
        self.terminalUpdate(self.string)


def start(nn):
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = UiMainWindow()
    ui.setupUi(MainWindow, nn)
    MainWindow.show()
    sys.exit(app.exec_())


start(nn.NeuralNetwork([784, 24, 24, 10], True, "data/weights[1].txt"))
