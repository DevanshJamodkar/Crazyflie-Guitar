import sys
import re
import os
import numpy as np
import cv2
import mss
import pyautogui
from PyQt5 import QtWidgets, QtCore, QtGui

# ---------------------
# CONFIG
# ---------------------
TEMPLATES_DIR = "templates"  # folder with 6 alphabet images
THRESHOLD = 0.75             # adjust after testing
USE_EDGES = True             # use edge-based matching for robustness

# Mapping: symbol -> command
COMMAND_MAP = {
    "A": "LEFT",
    "B": "RIGHT",
    "C": "UP",
    "D": "DOWN",
    "E": "FRONT",
    "F": "BACK"
}

# Mapping: command -> keypress
KEY_SEND_MAP = {
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "FRONT": "w",
    "BACK": "s"
}

# ---------------------
# Template Matching Utils
# ---------------------
def preprocess(img):
    """Convert to grayscale and optionally edges"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.equalizeHist(gray)  # normalize contrast
    if USE_EDGES:
        gray = cv2.Canny(gray, 50, 150)
    return gray

def load_templates(folder):
    templates = {}
    for fname in sorted(os.listdir(folder)):
        name, ext = os.path.splitext(fname)
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if USE_EDGES:
            img = cv2.Canny(img, 50, 150)
        templates[name] = img
    return templates

def match_symbol(frame_gray, templates):
    best_name = None
    best_score = -1.0

    for name, tmpl in templates.items():
        # resize template if needed to fit ROI
        if tmpl.shape[0] > frame_gray.shape[0] or tmpl.shape[1] > frame_gray.shape[1]:
            scale = min(frame_gray.shape[0]/tmpl.shape[0], frame_gray.shape[1]/tmpl.shape[1])
            new_w, new_h = int(tmpl.shape[1]*scale), int(tmpl.shape[0]*scale)
            tmpl_resized = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            tmpl_resized = tmpl

        res = cv2.matchTemplate(frame_gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_name = name

    if best_score >= THRESHOLD:
        return best_name, best_score
    return None, best_score

# ---------------------
# PyQt5 GUI
# ---------------------
class SelectionOverlay(QtWidgets.QWidget):
    region_selected = QtCore.pyqtSignal(tuple)  # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.setCursor(QtCore.Qt.CrossCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setOpacity(0.35)
        painter.fillRect(self.rect(), QtGui.QColor('black'))
        if not self.begin.isNull() and not self.end.isNull():
            painter.setOpacity(1.0)
            pen = QtGui.QPen(QtGui.QColor('red'), 2)
            painter.setPen(pen)
            rect = QtCore.QRect(self.begin, self.end)
            painter.drawRect(rect.normalized())

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end = event.pos()
        rect = QtCore.QRect(self.begin, self.end).normalized()
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        self.region_selected.emit((x, y, w, h))
        self.close()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen OCR → Commands (Template Matching)")
        self.resize(500, 220)
        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        hl = QtWidgets.QHBoxLayout()
        self.select_btn = QtWidgets.QPushButton("Select region")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        hl.addWidget(self.select_btn)
        hl.addWidget(self.start_btn)
        hl.addWidget(self.stop_btn)
        layout.addLayout(hl)

        # Interval
        interval_layout = QtWidgets.QHBoxLayout()
        interval_layout.addWidget(QtWidgets.QLabel("Interval (ms):"))
        self.interval_spin = QtWidgets.QSpinBox()
        self.interval_spin.setRange(100, 10000)
        self.interval_spin.setValue(800)
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addStretch()
        layout.addLayout(interval_layout)

        # Checkboxes
        cb_layout = QtWidgets.QHBoxLayout()
        self.send_keys_cb = QtWidgets.QCheckBox("Send keypress via pyautogui")
        self.send_keys_cb.setChecked(False)
        cb_layout.addWidget(self.send_keys_cb)
        cb_layout.addStretch()
        layout.addLayout(cb_layout)

        # Output displays
        self.last_text_label = QtWidgets.QLabel("Last detected symbol: ")
        self.last_cmd_label = QtWidgets.QLabel("Mapped command: ")
        layout.addWidget(self.last_text_label)
        layout.addWidget(self.last_cmd_label)

        # Preview
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setFixedSize(480, 100)
        self.preview_label.setStyleSheet("background: #222; border: 1px solid #555;")
        layout.addWidget(self.preview_label)

        # Signals
        self.select_btn.clicked.connect(self.open_selector)
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)

        # Vars
        self.region = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.capture_and_match)
        self.sct = mss.mss()
        self.templates = load_templates(TEMPLATES_DIR)

    def open_selector(self):
        self.overlay = SelectionOverlay()
        self.overlay.region_selected.connect(self.set_region)
        self.overlay.show()

    def set_region(self, region):
        self.region = region
        x, y, w, h = region
        self.last_text_label.setText(f"Last detected symbol: (region set {x},{y} {w}x{h})")

    def start_capture(self):
        if not self.region:
            QtWidgets.QMessageBox.warning(self, "No region", "Please select a region first.")
            return
        interval = self.interval_spin.value()
        self.timer.start(interval)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_btn.setEnabled(False)

    def stop_capture(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)

    def capture_and_match(self):
        x, y, w, h = self.region
        bbox = {"left": x, "top": y, "width": w, "height": h}
        s_img = self.sct.grab(bbox)
        frame = np.array(s_img)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        gray = preprocess(frame)
        symbol, score = match_symbol(gray, self.templates)

        if symbol:
            self.last_text_label.setText(f"Last detected symbol: {symbol} (score={score:.2f})")
            cmd = COMMAND_MAP.get(symbol, "(none)")
            self.last_cmd_label.setText(f"Mapped command: {cmd}")

            if self.send_keys_cb.isChecked() and cmd in KEY_SEND_MAP:
                try:
                    pyautogui.press(KEY_SEND_MAP[cmd])
                except Exception as e:
                    print("pyautogui error:", e)
        else:
            self.last_text_label.setText(f"Last detected symbol: (no match)")
            self.last_cmd_label.setText("Mapped command: (none)")

        # Preview image
        preview = cv2.resize(gray, (480, max(1, int(480 * gray.shape[0] / gray.shape[1]))))
        qimg = QtGui.QImage(preview.data, preview.shape[1], preview.shape[0], preview.strides[0],
                            QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), QtCore.Qt.KeepAspectRatio))


# ---------------------
# Main
# ---------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()