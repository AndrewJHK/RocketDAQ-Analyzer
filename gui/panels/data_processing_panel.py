from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout,
    QPushButton, QTextEdit, QCheckBox, QLineEdit, QFormLayout,
    QGroupBox, QFileDialog, QScrollArea, QListWidget, QRadioButton
)
from PyQt6.QtCore import QThreadPool, QTimer
from src.data_processing import DataProcessor
from src.processing_utils import logger, Worker, show_processing_dialog
import re


class DataProcessingPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.dataframes = {}
        self.processors = {}
        self.threadpool = QThreadPool()
        self.help_map = {
            "absolute": "Replace all data with absolute values",
            "drop": "Drop the data, based on a selected condition",
            "flip_sign": "Take content of each selected columns and change the sign + into -,- into + ",
            "normalize": "Take content of each selected column and perform a min-max value normalization of them",
            "offset": "Take content of each selected column and offset them by a value provided in parameters box",
            "rename": "Rename a selected column",
            "scale": "Take content of each selected column and scale them by a factor provided in parameters box",
            "sort": "Select only one column and sort the whole data by that specific column.In parameters specify if it should be ascending or descending by writing 'ascending=True/False'",
            "remove_negatives": "Replace all negative values with 0",
            "remove_positives": "Replace all positive values with 0",
            "rolling_mean": "Perform a rolling mean filter with a specified windows size in parameters",
            "rolling_median": "Perform a rolling median filter with a specified windows size in parameters",
            "threshold": "Replace all the values that exceed a provided value with that value",
            "wavelet_transform": " Perform a wavelet decomposition and recomposition with specified parameters"
        }

        layout = QVBoxLayout()

        self.file_selector = QComboBox()
        self.file_selector.currentIndexChanged.connect(self.update_columns)
        layout.addWidget(QLabel("Choose file:"))
        layout.addWidget(self.file_selector)

        side_layout = QHBoxLayout()

        # Columns
        self.column_box = QGroupBox("Columns")
        self.column_scroll = QScrollArea()
        self.column_widget = QWidget()
        self.column_layout = QVBoxLayout()
        self.column_widget.setLayout(self.column_layout)
        self.column_scroll.setWidgetResizable(True)
        self.column_scroll.setWidget(self.column_widget)
        column_box_layout = QVBoxLayout()
        column_box_layout.addWidget(self.column_scroll)
        self.column_box.setLayout(column_box_layout)
        side_layout.addWidget(self.column_box)

        # Operations
        self.operation_box = QGroupBox("Operations")
        self.operation_selector = QComboBox()
        self.operation_selector.addItems(["absolute", "drop", "flip_sign", "normalize","offset", "rename", "scale", "sort"])
        self.operation_selector.currentTextChanged.connect(self.toggle_drop_mode)
        self.operation_selector.currentTextChanged.connect(self.update_placeholder_operations)
        self.operation_selector.currentTextChanged.connect(self.update_operation_help)
        self.operation_param = QLineEdit()
        self.operation_param.setEnabled(False)
        self.operation_help = QLabel(
            "Replace all data with absolute values")
        self.operation_help.setWordWrap(True)
        self.apply_op_button = QPushButton("Apply operation")
        self.apply_op_button.clicked.connect(self.apply_operation)

        # Drop mode radios
        self.drop_columns_radio = QRadioButton("Drop by columns")
        self.drop_index_radio = QRadioButton("Drop by index range")
        self.drop_condition_radio = QRadioButton("Drop by condition (lambda)")
        self.drop_columns_radio.setChecked(True)
        self.drop_index_radio.hide()
        self.drop_columns_radio.hide()
        self.drop_condition_radio.hide()

        op_layout = QFormLayout()
        op_layout.addRow("Operation:", self.operation_selector)
        op_layout.addRow(self.drop_columns_radio)
        op_layout.addRow(self.drop_index_radio)
        op_layout.addRow(self.drop_condition_radio)
        op_layout.addRow("Parameter:", self.operation_param)
        op_layout.addRow(self.apply_op_button)
        op_layout.addRow(self.operation_help)
        self.operation_box.setLayout(op_layout)
        side_layout.addWidget(self.operation_box)

        # Filters
        self.filter_box = QGroupBox("Filters")
        self.filter_selector = QComboBox()
        self.filter_selector.addItems([
            "remove_negatives", "remove_positives", "rolling_mean", "rolling_median", "threshold",
            "wavelet_transform"])
        self.filter_selector.currentTextChanged.connect(self.update_placeholder_filters)
        self.filter_selector.currentTextChanged.connect(self.update_filter_help)
        self.filter_param = QLineEdit()
        self.filter_param.setEnabled(False)
        self.filter_help = QLabel("Replace all negative values with 0")
        self.filter_help.setWordWrap(True)
        self.add_filter_button = QPushButton("Add filter")
        self.add_filter_button.clicked.connect(self.add_filter)

        filter_layout = QFormLayout()
        filter_layout.addRow("Filter:", self.filter_selector)
        filter_layout.addRow(self.filter_param)
        filter_layout.addRow(self.add_filter_button)
        filter_layout.addRow(self.filter_help)
        self.filter_box.setLayout(filter_layout)
        side_layout.addWidget(self.filter_box)

        layout.addLayout(side_layout)

        # Filter queue
        self.queue_list = QListWidget()
        queue_box = QGroupBox("Filter queue")
        queue_layout = QVBoxLayout()
        queue_layout.addWidget(self.queue_list)
        queue_box.setLayout(queue_layout)
        layout.addWidget(queue_box)

        self.apply_filters_button = QPushButton("Apply filter")
        self.apply_filters_button.clicked.connect(self.apply_filters)
        layout.addWidget(self.apply_filters_button)

        self.save_button = QPushButton("Save to file")
        self.save_button.clicked.connect(self.save_to_file)
        layout.addWidget(self.save_button)

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.status_log)

        self.setLayout(layout)

    def toggle_drop_mode(self, operation):
        is_drop = operation == "drop"
        self.drop_columns_radio.setVisible(is_drop)
        self.drop_index_radio.setVisible(is_drop)
        self.drop_condition_radio.setVisible(is_drop)

    def update_placeholder_operations(self, operation):
        placeholders = {
            "normalize": "",
            "scale": "100",
            "offset": "10",
            "flip_sign": "",
            "absolute": "",
            "sort": "ascending=True/False",
            "drop": "e.g. 100,200 or row['column_name'] > 0",
            "rename": "new_name"
        }
        match operation:
            case "normalize" | "flip_sign" | "absolute":
                self.operation_param.setText(placeholders.get(operation, ""))
                self.operation_param.setEnabled(False)
            case _:
                self.operation_param.setText(placeholders.get(operation, ""))
                self.operation_param.setEnabled(True)

    def update_placeholder_filters(self, operation):
        placeholders = {
            "remove_negatives": "",
            "remove_positives": "",
            "rolling_mean": "window=2137",
            "rolling_median": "window=2137",
            "threshold": "threshold=2137",
            "wavelet_transform": "wavelet_name=coif5,level=10,threshold_mode=soft"
        }
        match operation:
            case "remove_negatives" | "remove_positives":
                self.filter_param.setText(placeholders.get(operation, ""))
                self.filter_param.setEnabled(False)
            case _:
                self.filter_param.setText(placeholders.get(operation, ""))
                self.filter_param.setEnabled(True)

    def update_filter_help(self):
        self.filter_help.setText(self.help_map.get(self.filter_selector.currentText(), ""))

    def update_operation_help(self):
        self.operation_help.setText(self.help_map.get(self.operation_selector.currentText(), ""))

    def add_dataframe(self, file_path, wrapper):
        self.dataframes[file_path] = wrapper
        self.processors[file_path] = DataProcessor(wrapper)
        self.file_selector.addItem(file_path)

    def remove_dataframe(self, file_path):
        if file_path in self.dataframes:
            del self.dataframes[file_path]
        if file_path in self.processors:
            del self.processors[file_path]
        index = self.file_selector.findText(file_path)
        if index >= 0:
            self.file_selector.removeItem(index)
        self.update_columns()

    def update_columns(self):
        self.clear_column_checkboxes()
        current_path = self.file_selector.currentText()
        if current_path and current_path in self.dataframes:
            df = self.dataframes[current_path].get_dataframe()
            for col in df.columns:
                checkbox = QCheckBox(col)
                self.column_layout.addWidget(checkbox)

    def clear_column_checkboxes(self):
        while self.column_layout.count():
            child = self.column_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_selected_columns(self):
        return [self.column_layout.itemAt(i).widget().text()
                for i in range(self.column_layout.count())
                if self.column_layout.itemAt(i).widget().isChecked()]

    def log(self, message, log_type):
        match log_type:
            case "DEBUG":
                logger.debug(message)
            case "INFO":
                logger.info(message)
            case "ERROR":
                logger.error(message)
        self.status_log.append(message)

    def apply_operation(self):
        def task():
            file_path = self.file_selector.currentText()
            columns = self.get_selected_columns()
            operation = self.operation_selector.currentText()
            param = self.operation_param.text()
            success = False

            if not file_path or not operation:
                self.log("Choose file and operation.", "INFO")
                return

            processor = self.processors.get(file_path)
            try:
                match operation:
                    case "normalize":
                        processor.normalize_columns(columns)
                        success = True
                    case "scale":
                        processor.scale_columns(columns, float(param))
                        success = True
                    case "offset":
                        processor.offset(columns, float(param))
                        success = True
                    case "flip_sign":
                        processor.flip_column_sign(columns)
                        success = True
                    case "rename":
                        if len(columns) == 1:
                            processor.rename_column(columns[0], str(param))
                            QTimer.singleShot(200, self.update_columns)
                            success = True
                        else:
                            self.log("You may rename only one column at a time", "ERROR")
                    case "absolute":
                        processor.absolute(columns)
                        success = True
                    case "sort":
                        if len(columns)<=0:
                            processor.sort_data(columns[0], ascending=True)
                            success = True
                        else:
                            self.log("You may sort by only one column", "ERROR")
                    case "drop":
                        if self.drop_columns_radio.isChecked():
                            processor.drop_data(columns=columns)
                            QTimer.singleShot(200, self.update_columns)
                            success = True
                        elif self.drop_index_radio.isChecked():
                            start_end = param.split(',')
                            if len(start_end) == 2:
                                start, end = int(start_end[0]), int(start_end[1])
                                processor.drop_data(row_range=(start, end))
                                success = True
                        elif self.drop_condition_radio.isChecked():
                            try:
                                processor.drop_data(row_condition=lambda row: eval(param))
                                success = True
                            except Exception as e:
                                self.log(f"Invalid lambda: {e}", "ERROR")
                if success:
                    self.log(f"Applied operation: {operation} on columns: {columns}", "INFO")
            except Exception as e:
                self.log(f"Error during operation: {e}", "ERROR")

        show_processing_dialog(self, self.threadpool, Worker(task))

    def add_filter(self):
        file_path = self.file_selector.currentText()
        columns = self.get_selected_columns()
        filter_type = self.filter_selector.currentText()
        raw_params = self.filter_param.text()
        processor = self.processors.get(file_path)

        try:
            params = dict(p.split('=') for p in raw_params.split(',') if '=' in p)
            number_pattern = re.compile(r"^-?\d+(\.\d+)?$")
            for k in params:
                value = params[k]
                if number_pattern.fullmatch(value):
                    if k == "level":
                        params[k] = int(value)
                    else:
                        params[k] = float(value)
                else:
                    params[k] = value
            processor.add_filter(columns, filter_type, **params)
            self.queue_list.addItem(f"{filter_type} on {columns} with {params}")
            self.log(f"Added filter {filter_type} for {columns}", "INFO")
        except Exception as e:
            self.log(f"Error adding filter: {e}", "ERROR")

    def apply_filters(self):
        try:
            def task():
                file_path = self.file_selector.currentText()
                processor = self.processors.get(file_path)
                processor.queue_filters()
                self.log("Applied all queued filters", "INFO")

            show_processing_dialog(self, self.threadpool, Worker(task))
        except Exception as e:
            self.log(f"Error during filter aplication:{e}", "ERROR")

    def save_to_file(self):
        file_path = self.file_selector.currentText()
        processor = self.processors.get(file_path)
        if processor:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save as", filter="CSV Files (*.csv)")
            if save_path:
                try:
                    processor.save_data(save_path)
                    self.log(f"Data saved to: {save_path}", "INFO")
                except Exception as e:
                    self.log(f"Error during saving data: {e}", "ERROR")
