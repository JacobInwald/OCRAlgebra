from PIL import Image
from PyQt5 import *

class gui:

    def __init__(self, size, path):
        width, height = size
        self.startDisplay()
        self.background = Image.open(path)

    def startDisplay(self):
        # code to initialise the display
        return