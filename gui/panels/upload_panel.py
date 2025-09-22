from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QHBoxLayout, QListWidget, QTextEdit,
    QRadioButton, QButtonGroup
)
import os
import json
from PyQt6.QtCore import QThreadPool
from src.data_processing import DataFrameWrapper
from src.json_parser import JSONParser
from src.processing_utils import logger, Worker, show_processing_dialog


class UploadPanel(QWidget):
    def __init__(self, add_callback=None, add_file_widget=None):
        super().__init__()
        self.add_callback = add_callback
        self.add_file_widget = add_file_widget
        self.loaded_files = []
        self.interpolated = True
        self.threadpool = QThreadPool()

        layout = QVBoxLayout()

        self.csv_button = QPushButton("Load CSV")
        self.csv_button.clicked.connect(self.load_csv)

        self.json_button = QPushButton("Convert JSON")
        self.json_button.clicked.connect(self.load_json)

        self.radio_group = QButtonGroup(self)
        self.interpolated_radio = QRadioButton("Interpolated")
        self.none_filled_radio = QRadioButton("None filled")
        self.interpolated_radio.setChecked(True)
        self.radio_group.addButton(self.interpolated_radio)
        self.radio_group.addButton(self.none_filled_radio)

        self.interpolated_radio.toggled.connect(self.update_interpolation_mode)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.interpolated_radio)
        radio_layout.addWidget(self.none_filled_radio)

        self.file_log_layout = QHBoxLayout()

        self.file_layout = QVBoxLayout()
        self.file_layout.addWidget(QLabel("Loaded files:"))

        self.file_list = QListWidget()
        self.file_layout.addWidget(self.file_list)

        self.file_log_layout.addLayout(self.file_layout)

        self.log_layout = QVBoxLayout()
        self.log_layout.addWidget(QLabel("Logs:"))

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setPlaceholderText("Logs will appear here...")
        self.log_layout.addWidget(self.status_log)

        self.file_log_layout.addLayout(self.log_layout)

        layout.addWidget(self.csv_button)
        layout.addWidget(self.json_button)
        layout.addLayout(radio_layout)
        layout.addLayout(self.file_log_layout)

        self.setLayout(layout)

    def update_interpolation_mode(self):
        self.interpolated = self.interpolated_radio.isChecked()
        logger.info(f"JSON conversion mode: {'Interpolated' if self.interpolated else 'None filled'}")

    def log(self, message, log_type):
        match log_type:
            case "DEBUG":
                logger.debug(message)
            case "INFO":
                logger.info(message)
            case "ERROR":
                logger.error(message)
        self.status_log.append(message)

    def load_csv(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose CSV files", filter="CSV Files (*.csv)")
        if not files:
            return
        for file_path in files:
            try:
                def task(path=file_path, signals=None):
                    if path and path not in self.loaded_files:
                        wrapper = DataFrameWrapper(path)
                        self.loaded_files.append(path)
                        self.file_list.addItem(path)
                        if self.add_callback:
                            self.add_callback(path, wrapper)
                        if signals:
                            signals.file_ready.emit(path)
                        self.log(f"Loaded CSV file: {path}", "INFO")

                worker = Worker(task)
                worker.signals.file_ready.connect(self.add_file_widget)
                worker.fn = lambda: task(signals=worker.signals)
                show_processing_dialog(self, self.threadpool, worker)
            except Exception as e:
                self.log(f"Error while loading CSV:{file_path}: {e}", "ERROR")

    def remove_dataframe(self, file_path):
        if file_path in self.loaded_files:
            self.loaded_files.remove(file_path)

            for i in range(self.file_list.count()):
                if self.file_list.item(i).text() == file_path:
                    self.file_list.takeItem(i)
                    break

            self.log(f"File removed: {file_path}", "INFO")

    def load_json(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose JSON files", filter="JSON Files (*.json)")
        if not files:
            return

        def task():
            for file_path in files:
                base_path = os.path.splitext(file_path)[0]
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        parser = JSONParser(data, base_path, interpolated=self.interpolated)
                        generated_paths = parser.json_to_csv()
                        for path in generated_paths:
                            if self.add_file_widget:
                                self.add_file_widget(path)
                        self.log(f"Converted JSON to CSVs: {base_path}", "INFO")
                except Exception as e:
                    self.log(f"JSON conversion error: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))
