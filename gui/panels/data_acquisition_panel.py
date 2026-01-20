from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QHBoxLayout, QListWidget, QTextEdit,
    QRadioButton, QButtonGroup, QLineEdit, QComboBox, QTimeEdit, QDateEdit
)
import os
import json
from PyQt6.QtCore import QThreadPool, QDate, QTime
from src.data_processing import DataFrameWrapper
from src.data_acquisition import DATAParser, MongoDBDataRetriever
from src.processing_utils import logger, Worker, show_processing_dialog
from datetime import datetime


class UploadPanel(QWidget):
    def __init__(self, add_callback=None, add_file_widget=None):
        super().__init__()
        self.add_callback = add_callback
        self.add_file_widget = add_file_widget
        self.loaded_files = []
        self.interpolated = True
        self.threadpool = QThreadPool()

        # Main layout
        layout = QVBoxLayout()

        # Upper layout
        self.upper_layout = QHBoxLayout()

        # Load layout
        self.load_layout = QVBoxLayout()

        self.bson_button = QPushButton("Convert BSON")
        self.bson_button.clicked.connect(self.load_bson)
        self.load_layout.addWidget(self.bson_button)

        self.json_button = QPushButton("Convert JSON")
        self.json_button.clicked.connect(self.load_json)
        self.load_layout.addWidget(self.json_button)

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
        self.load_layout.addLayout(radio_layout)

        self.csv_button = QPushButton("Load CSV")
        self.csv_button.clicked.connect(self.load_csv)
        self.load_layout.addWidget(self.csv_button)

        self.upper_layout.addLayout(self.load_layout)

        # Retrieve layout
        self.retrieve_layout = QVBoxLayout()
        self.retrieve_widget = QWidget()
        self.retrieve_widget.setLayout(self.retrieve_layout)
        self.retrieve_widget.setMaximumWidth(750)
        self.upper_layout.addWidget(self.retrieve_widget)

        # Database input layout
        db_address_layout = QHBoxLayout()

        port_label = QLabel("IP:")
        db_address_layout.addWidget(port_label)
        self.ip_address_box = QLineEdit("localhost")
        db_address_layout.addWidget(self.ip_address_box)

        port_label = QLabel("Port:")
        db_address_layout.addWidget(port_label)
        self.port_box = QLineEdit("27017")
        db_address_layout.addWidget(self.port_box)

        self.retrieve_layout.addLayout(db_address_layout)

        self.data_retrieve_button = QPushButton("Reload available data")
        self.data_retrieve_button.clicked.connect(self.retrieve_bson)

        self.retrieve_layout.addWidget(self.data_retrieve_button)

        # Available databases combobox
        self.retrieved_databases = QComboBox()
        self.retrieved_databases.currentTextChanged.connect(self.reload_collections)
        self.retrieve_layout.addWidget(self.retrieved_databases)

        # Available collections combobox
        self.retrieved_collections = QComboBox()
        self.retrieved_collections.currentTextChanged.connect(self.read_collection)
        self.retrieve_layout.addWidget(self.retrieved_collections)

        # Amount of docs layout
        doc_count_layout = QHBoxLayout()
        count_label = QLabel("Number of docs in that collection: ")
        doc_count_layout.addWidget(count_label)

        self.count_text = QLineEdit("0")
        self.count_text.setReadOnly(True)
        doc_count_layout.addWidget(self.count_text)

        self.retrieve_layout.addLayout(doc_count_layout)

        # Retrieve mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Download mode:"))

        self.mode_group = QButtonGroup(self)
        self.mode_by_index = QRadioButton("By doc range")
        self.mode_by_time = QRadioButton("By time range")
        self.mode_by_index.setChecked(True)

        self.mode_by_index.toggled.connect(self.update_download_mode)
        self.mode_by_time.toggled.connect(self.update_download_mode)

        self.mode_group.addButton(self.mode_by_index)
        self.mode_group.addButton(self.mode_by_time)

        mode_layout.addWidget(self.mode_by_index)
        mode_layout.addWidget(self.mode_by_time)
        self.retrieve_layout.addLayout(mode_layout)

        # Doc range to download layout

        doc_range_layout = QHBoxLayout()
        range_label = QLabel("Range of docs to download")
        doc_range_layout.addWidget(range_label)

        self.doc_range_input = QLineEdit("")
        self.doc_range_input.setPlaceholderText("ex1. 0 ex2. 0,2137 ex3. 420")

        doc_range_layout.addWidget(self.doc_range_input)

        self.retrieve_layout.addLayout(doc_range_layout)

        self.download_collection = QPushButton("Download collection")
        self.download_collection.clicked.connect(self.download_bson)

        # Time range layout
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(QLabel("Time range:"))

        # Start
        start_layout = QVBoxLayout()
        start_layout.addWidget(QLabel("Start date / time"))
        self.start_date_edit = QDateEdit(QDate(2001,9,11))
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_time_edit = QTimeEdit(QTime(16,20))
        self.start_time_edit.setDisplayFormat("HH:mm")
        start_layout.addWidget(self.start_date_edit)
        start_layout.addWidget(self.start_time_edit)

        # Stop
        stop_layout = QVBoxLayout()
        stop_layout.addWidget(QLabel("End date / time"))
        self.end_date_edit = QDateEdit(QDate(2001,9,11))
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_time_edit = QTimeEdit(QTime(21,37))
        self.end_time_edit.setDisplayFormat("HH:mm")
        stop_layout.addWidget(self.end_date_edit)
        stop_layout.addWidget(self.end_time_edit)

        time_range_layout.addLayout(start_layout)
        time_range_layout.addLayout(stop_layout)

        self.retrieve_layout.addLayout(time_range_layout)

        self.retrieve_layout.addWidget(self.download_collection)

        # Logging layout
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

        # Download mode startup
        self.update_download_mode()

        # Layout combination
        layout.addLayout(self.upper_layout)
        layout.addLayout(self.file_log_layout)

        self.setLayout(layout)

        # Data Retriever
        try:
            self.data_retriever = MongoDBDataRetriever("localhost", 27017)
            self.log(f"Succesfully setup a data retriever", "INFO")
        except Exception as e:
            self.log(f"Couldn't setup a data retriever due to an error: {e}", "ERROR")

    def update_interpolation_mode(self):
        self.interpolated = self.interpolated_radio.isChecked()
        self.log(f"JSON conversion mode: {'Interpolated' if self.interpolated else 'None filled'}", "INFO")

    def update_download_mode(self):
        by_index = self.mode_by_index.isChecked()

        # doc range widgets
        self.doc_range_input.setEnabled(by_index)

        # time range widgets
        self.start_date_edit.setEnabled(not by_index)
        self.start_time_edit.setEnabled(not by_index)
        self.end_date_edit.setEnabled(not by_index)
        self.end_time_edit.setEnabled(not by_index)

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
                        parser = DATAParser(data, base_path, interpolated=self.interpolated)
                        generated_paths = parser.json_to_csv()
                        for path in generated_paths:
                            if self.add_file_widget:
                                self.add_file_widget(path)
                        self.log(f"Converted JSON to CSVs: {base_path}", "INFO")
                except Exception as e:
                    self.log(f"JSON conversion error: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))

    def retrieve_bson(self):
        ip = self.ip_address_box.text().strip()
        port_text = self.port_box.text().strip()

        def task():
            try:
                port = int(port_text)
                self.log(f"Connecting to MongoDB at {ip}:{port} ...", "INFO")
                self.data_retriever.change_client(ip, port)
                self.data_retriever.retrieve_databases()
                dbs = list(self.data_retriever.get_databases().keys())

                self.retrieved_databases.clear()
                self.retrieved_collections.clear()
                for db in dbs:
                    self.retrieved_databases.addItem(db)

                if dbs:
                    first_db = dbs[0]
                    current_db = self.retrieved_databases.currentText() or first_db
                    self.retrieved_collections.addItems(self.data_retriever.get_collections_in_database(current_db))
                    self.log(f"Databases reloaded ({len(dbs)} found).", "INFO")
                else:
                    self.log("No databases found.", "INFO")
            except Exception as e:
                self.log(f"Failed to reload databases: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))

    def reload_collections(self, db):
        self.retrieved_collections.clear()
        db = self.retrieved_databases.currentText()
        if not db:
            return
        self.retrieved_collections.addItems(self.data_retriever.get_collections_in_database(db))

    def read_collection(self, collection):
        try:
            amount = self.data_retriever.get_doc_amount(self.retrieved_databases.currentText(),
                                                        collection)
            self.count_text.setText(str(amount))
        except Exception as e:
            self.log(f"Error while reading collection size: {e}", "ERROR")
            self.count_text.setText("0")

    def download_bson(self):
        database = self.retrieved_databases.currentText()
        collection = self.retrieved_collections.currentText()

        if not database or not collection:
            self.log("No database/collection selected.", "ERROR")
            return

        if self.mode_by_index.isChecked():
            range_text = self.doc_range_input.text().strip()
            try:
                start_range, end_range = 0, None
                if "," in range_text:
                    start_range, end_range = map(int, range_text.split(",", 1))
                elif range_text:
                    start_range = int(range_text)
            except Exception as e:
                self.log(f"Invalid range: '{range_text}'. Use 'start' or 'start,end'. ({e})", "ERROR")
                return

            try:
                os.makedirs("data", exist_ok=True)
            except Exception as e:
                self.log(f"Cannot create data directory: {e}", "ERROR")
                return

            suffix = end_range if end_range is not None else "end"
            outfile = os.path.join("data", f"{database}_{collection}_range_{start_range}_{suffix}.bson")

            def task():
                try:
                    self.log(f"Downloading {database}.{collection} "f"[{start_range}:{suffix}] → {outfile}", "INFO")
                    self.data_retriever.retrieve_bson_range(database, collection, outfile, start_range, end_range)
                    self.log(f"Saved BSON to {outfile}", "INFO")
                except Exception as e:
                    self.log(f"Error while downloading BSON: {e}", "ERROR")

            show_processing_dialog(self, self.threadpool, Worker(task))
            return

        try:
            start_dt = datetime.combine(self.start_date_edit.date().toPyDate(), self.start_time_edit.time().toPyTime())
            end_dt = datetime.combine(self.end_date_edit.date().toPyDate(), self.end_time_edit.time().toPyTime())
        except Exception as e:
            self.log(f"Invalid date/time selection: {e}", "ERROR")
            return

        if end_dt <= start_dt:
            self.log("End time must be after start time.", "ERROR")
            return

        try:
            os.makedirs("data", exist_ok=True)
        except Exception as e:
            self.log(f"Cannot create data directory: {e}", "ERROR")
            return

        start_str = start_dt.strftime("%Y%m%dT%H%M")
        end_str = end_dt.strftime("%Y%m%dT%H%M")
        outfile = os.path.join("data", f"{database}_{collection}_time_{start_str}_{end_str}.bson")

        def task():
            try:
                self.log(f"Downloading {database}.{collection} "f"[{start_dt}_{end_dt}]_{outfile}", "INFO")
                self.data_retriever.retrieve_bson_time_range(database, collection, outfile, start_dt, end_dt)
                self.log(f"Saved BSON to {outfile}", "INFO")
            except Exception as e:
                self.log(f"Error while downloading BSON by time: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))

    def load_bson(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose BSON files", filter="BSON Files (*.bson)")
        if not files:
            return

        def task():
            try:
                DATAParser.bson_files_to_json(files)
                for f in files:
                    base, _ = os.path.splitext(f)
                    self.log(f"Converted BSON to JSON: {base}.json", "INFO")
            except Exception as e:
                self.log(f"BSON conversion error: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))
