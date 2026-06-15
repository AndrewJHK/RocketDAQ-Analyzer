import logging
import traceback

from PyQt6.QtCore import Qt, QRunnable, QTimer, pyqtSignal, QObject, pyqtSlot
from PyQt6.QtWidgets import QProgressDialog, QMessageBox

logger = logging.getLogger("postprocessing-app")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(file_handler)

logger.propagate = False


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    log = pyqtSignal(str, str)
    progress = pyqtSignal(int)
    file_ready = pyqtSignal(str)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Allow task to emit safely
        self.kwargs["signals"] = self.signals

    @pyqtSlot()
    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(res)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


def show_processing_dialog(parent, threadpool, worker: Worker):
    dialog = QProgressDialog("Processing, please wait...", None, 0, 0, parent)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setWindowTitle("Processing...")

    def _on_error(tb: str):
        dialog.close()
        logger.error(tb)
        QMessageBox.critical(parent, "Error", tb)

    worker.signals.finished.connect(dialog.close)
    worker.signals.error.connect(_on_error)

    QTimer.singleShot(0, lambda: threadpool.start(worker))
    dialog.exec()
