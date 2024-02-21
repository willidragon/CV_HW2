# graffiti.py
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QMouseEvent, QPen, QColor, QPalette
from PyQt5.QtCore import Qt, QPoint

class GraffitiBoard(QWidget):
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.last_point = QPoint()
        self.lines = []  # This will store the lines to draw
        self.initUI()

    def initUI(self):
        # Set the background to black using a palette
        palette = self.palette()  # Get the current palette
        palette.setColor(QPalette.ColorRole.Window, QColor('black'))  # Set the window (background) color to black
        self.setPalette(palette)  # Apply the palette to the widget

        # Set the pen color to white for drawing
        self.pen = QPen(QColor('white'), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.setAutoFillBackground(True)  # Ensure the background color fills the widget


    def mousePressEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()  # Use pos() instead of position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            current_point = event.pos()  # Use pos() instead of position()
            self.lines.append((self.last_point, current_point))
            self.last_point = current_point
            self.update()  # Schedule a repaint

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.pen)
        for line in self.lines:
            painter.drawLine(line[0], line[1])

    def resetCanvas(self):
        self.lines = []
        self.update()