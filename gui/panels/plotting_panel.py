from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout,
    QPushButton, QLineEdit, QFormLayout,
    QGroupBox, QScrollArea, QFrame, QRadioButton, QButtonGroup, QCheckBox, QSizePolicy
)

from src.data_processing import DataProcessor
from src.plotter import Plotter
from src.processing_utils import logger


class PlotColumnWidget(QWidget):
    def __init__(self, columns, remove_callback=None, is_scatter=False):
        super().__init__()
        self.layout = QFormLayout()

        self.channel_input = QComboBox()
        self.channel_input.addItems(columns)

        self.label_input = QLineEdit("Placeholder")

        self.color_input = QComboBox()
        self.color_input.addItems(['blue', 'red', 'green', 'cyan', 'purple', 'olive', 'pink', 'gray', 'brown'])

        self.transparency_input = QLineEdit("1")

        self.y_axis_input = QComboBox()
        self.y_axis_input.addItems(["y1", "y2"])

        self.x_column_input = QComboBox()
        self.x_column_input.addItems(columns)
        for column in columns:
            if "timestamp" in column:
                self.x_column_input.setCurrentText(column)

        self.size_input = QLineEdit()
        self.size_input.setDisabled(not is_scatter)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(lambda: remove_callback(self) if remove_callback else None)

        self.layout.addRow("Data:", self.channel_input)
        self.layout.addRow("Label:", self.label_input)
        self.layout.addRow("Color:", self.color_input)
        self.layout.addRow("Transparency:", self.transparency_input)
        self.layout.addRow("Y Axis:", self.y_axis_input)
        self.layout.addRow("X Axis:", self.x_column_input)
        self.layout.addRow("Dot Size:", self.size_input)
        self.layout.addRow(self.remove_button)

        self.setLayout(self.layout)

    def set_scatter_mode(self, enabled: bool):
        self.size_input.setDisabled(not enabled)

    def get_config(self):
        return {
            "channel": self.channel_input.currentText(),
            "label": self.label_input.text(),
            "color": self.color_input.currentText(),
            "alpha": float(self.transparency_input.text()) if self.transparency_input.text() else 1.0,
            "y_axis": self.y_axis_input.currentText(),
            "x_column": self.x_column_input.currentText(),
            "size": int(self.size_input.text()) if self.size_input.text() else 1,
        }


class PlottingPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.dataframes = {}
        self.secondary_db_offset = 0
        self.db1_columns = []
        self.db2_columns = []

        layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        left_top = QVBoxLayout()
        left_top.addWidget(QLabel("Title: "))
        self.plot_name = QLineEdit("Plot Title")
        left_top.addWidget(self.plot_name)

        self.x_axis_label = QLineEdit("Time")
        self.y1_axis_label = QLineEdit("Y1 Axis")
        self.y2_axis_label = QLineEdit("Y2 Axis")
        left_top.addWidget(QLabel("X Axis Label:"))
        left_top.addWidget(self.x_axis_label)
        left_top.addWidget(QLabel("Y1 Axis Label:"))
        left_top.addWidget(self.y1_axis_label)
        left_top.addWidget(QLabel("Y2 Axis Label:"))
        left_top.addWidget(self.y2_axis_label)

        self.offset_input = QLineEdit("0")
        left_top.addWidget(QLabel("Offset (ms):"))
        left_top.addWidget(self.offset_input)

        right_top = QVBoxLayout()
        self.db1_selector = QComboBox()
        self.db1_selector.currentIndexChanged.connect(lambda: self.refresh_channel_box("db1"))
        right_top.addWidget(QLabel("Select DB1:"))
        right_top.addWidget(self.db1_selector)

        self.db2_selector = QComboBox()
        self.db2_selector.currentIndexChanged.connect(lambda: self.refresh_channel_box("db2"))
        right_top.addWidget(QLabel("Select DB2:"))
        right_top.addWidget(self.db2_selector)
        self.convert_epoch = QComboBox()
        self.convert_epoch.addItems(["none", "seconds", "miliseconds"])
        right_top.addWidget(QLabel("Convert epoch to:"))
        right_top.addWidget(self.convert_epoch)

        plot_type_layout = QHBoxLayout()
        self.radio_line = QRadioButton("Line Plot")
        self.radio_scatter = QRadioButton("Scatter Plot")
        self.grid_type = QCheckBox("Precise Grid")
        self.radio_line.setChecked(True)
        self.plot_type_group = QButtonGroup()
        self.plot_type_group.addButton(self.radio_line)
        self.plot_type_group.addButton(self.radio_scatter)
        plot_type_layout.addWidget(QLabel("Plot Type:"))
        plot_type_layout.addWidget(self.radio_line)
        plot_type_layout.addWidget(self.radio_scatter)
        plot_type_layout.addWidget(self.grid_type)
        right_top.addLayout(plot_type_layout)

        self.radio_line.toggled.connect(self.toggle_size_mode)
        self.radio_scatter.toggled.connect(self.toggle_size_mode)

        self.start_offset = QLineEdit()
        self.start_offset.setReadOnly(True)
        right_top.addWidget(QLabel("Start offset:"))
        right_top.addWidget(self.start_offset)

        sync_row = QHBoxLayout()
        self.sync_col1 = QComboBox()
        self.sync_col1.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.sync_col2 = QComboBox()
        self.sync_col2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sync_row.addWidget(QLabel("Sync DB1:"))
        sync_row.addWidget(self.sync_col1)
        sync_row.addWidget(QLabel("DB2:"))
        sync_row.addWidget(self.sync_col2)
        right_top.addLayout(sync_row)

        self.sync_button = QPushButton("Sync DBs")
        self.sync_button.clicked.connect(self.sync_databases)
        right_top.addWidget(self.sync_button)

        top_layout.addLayout(left_top)
        top_layout.addLayout(right_top)
        layout.addLayout(top_layout)

        self.db1_box = self.create_database_box("db1")
        self.db2_box = self.create_database_box("db2")
        db_layout = QHBoxLayout()
        db_layout.addWidget(self.db1_box)
        db_layout.addWidget(self.db2_box)
        layout.addLayout(db_layout, stretch=2)

        self.h_lines_box = self.create_lines_box("horizontal")
        self.v_lines_box = self.create_lines_box("vertical")
        lines_layout = QHBoxLayout()
        lines_layout.addWidget(self.h_lines_box)
        lines_layout.addWidget(self.v_lines_box)
        layout.addLayout(lines_layout, stretch=1)

        self.generate_button = QPushButton("Generate Plot")
        self.generate_button.clicked.connect(self.generate_plot)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)

    def toggle_size_mode(self):
        is_scatter = self.radio_scatter.isChecked()
        for widget in self.db1_columns + self.db2_columns:
            widget.set_scatter_mode(is_scatter)

    def create_database_box(self, db_key):
        box = QGroupBox(f"{db_key.upper()} Channels")
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        channel_layout = QVBoxLayout()
        container.setLayout(channel_layout)
        scroll.setWidget(container)

        add_button = QPushButton("Add Plot Column")
        add_button.clicked.connect(lambda: self.add_plot_column(db_key))

        layout.addWidget(scroll)
        layout.addWidget(add_button)

        box.setLayout(layout)
        box.channel_layout = channel_layout
        return box

    def add_plot_column(self, db_key):
        db_selector = self.db1_selector if db_key == "db1" else self.db2_selector
        db_path = db_selector.currentText()

        if db_path not in self.dataframes:
            return

        df = self.dataframes[db_path].get_dataframe()
        columns = df.columns

        is_scatter = self.radio_scatter.isChecked()

        widget = PlotColumnWidget(
            columns=columns,
            is_scatter=is_scatter,
            remove_callback=lambda w: self.remove_plot_column(db_key, w)
        )

        if db_key == "db1":
            self.db1_columns.append(widget)
            self.db1_box.channel_layout.addWidget(widget)
        else:
            self.db2_columns.append(widget)
            self.db2_box.channel_layout.addWidget(widget)

    def remove_plot_column(self, db_key, widget):
        if db_key == "db1" and widget in self.db1_columns:
            self.db1_columns.remove(widget)
        elif db_key == "db2" and widget in self.db2_columns:
            self.db2_columns.remove(widget)
        widget.setParent(None)
        widget.deleteLater()

    def generate_plot(self):
        db1_path = self.db1_selector.currentText()
        db2_path = self.db2_selector.currentText()
        try:
            offset_val = float(self.offset_input.text())
        except ValueError:
            offset_val = 0
        plot_type = "line" if self.radio_line.isChecked() else "scatter"
        config = {
            "plot_settings": {
                "title": self.plot_name.text(),
                "type": plot_type,
                "precise_grid": self.grid_type.isChecked(),
                "convert_epoch": self.convert_epoch.currentText(),
                "offset": offset_val,
                "secondary_db_offset": self.secondary_db_offset,
                "x_axis_label": self.x_axis_label.text(),
                "y_axis_labels": {
                    "y1": self.y1_axis_label.text(),
                    "y2": self.y2_axis_label.text()
                },
                "horizontal_lines": self.collect_lines(self.h_lines_box.lines_layout),
                "vertical_lines": self.collect_lines(self.v_lines_box.lines_layout)
            },
            "databases": {}
        }

        for db_key, column_list in [("db1", self.db1_columns), ("db2", self.db2_columns)]:
            if not column_list:
                continue
            db_config = {"channels": {}}
            for widget in column_list:
                try:
                    cfg = widget.get_config()
                    db_config["channels"][cfg["channel"]] = cfg
                except Exception as e:
                    self.log(f"Error reading plot column config: {e}", "ERROR")
            if db_config["channels"]:
                config["databases"][db_key] = db_config

        if not config["databases"]:
            self.log("No channels configured.", "INFO")
            return

        selected_dataframes = {}
        if db1_path in self.dataframes:
            selected_dataframes["db1"] = self.dataframes[db1_path]
        if db2_path in self.dataframes:
            selected_dataframes["db2"] = self.dataframes[db2_path]

        plotter = Plotter(config_dict=config, dataframe_map=selected_dataframes, plots_folder_path="plots")
        try:
            plotter.plot()
            self.log("Plot generated successfully.", "INFO")
        except Exception as e:
            self.log(f"Error during plot generation: {e}", "ERROR")

    def sync_databases(self):
        db1_path = self.db1_selector.currentText()
        db2_path = self.db2_selector.currentText()
        col1 = self.sync_col1.currentText()
        col2 = self.sync_col2.currentText()
        split = db1_path.split("/")
        db1_name=split[-1]
        if db1_path not in self.dataframes or db2_path not in self.dataframes:
            self.log("Both DBs must be selected for syncing.", "INFO")
            return

        try:
            dp1 = DataProcessor(self.dataframes[db1_path])
            dp2 = DataProcessor(self.dataframes[db2_path])
            df1 = dp1.get_processed_data()
            df2 = dp2.get_processed_data()

            idx1, val1 = dp1.find_index_where_max("header.timestamp_epoch", col1)
            _, val2 = dp2.find_index_where_max("header.timestamp_epoch", col2)
            self.secondary_db_offset = val1 - val2
            self.log(f"Syncing DB2 by offset {self.secondary_db_offset} based on max of {col1} and {col2}", "INFO")

            df2[f"header.timestamp_epoch_syncedwith_{db1_name}"] = df2["header.timestamp_epoch"].compute() + self.secondary_db_offset
            dp2.df_wrapper.update_dataframe(df2)

            first_value = df1['header.timestamp_epoch'].compute().iloc[0]
            start_offset = first_value - val1
            self.start_offset.setText(f"Start offset = {start_offset}")

        except Exception as e:
            self.log(f"Error during DB sync: {e}", "ERROR")

    def create_lines_box(self, line_type):
        box = QGroupBox(f"{line_type.title()} Lines")
        layout = QVBoxLayout()
        box.setLayout(layout)
        box.setMaximumHeight(250)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        lines_layout = QVBoxLayout()
        container.setLayout(lines_layout)
        scroll.setWidget(container)

        add_button = QPushButton(f"Add {line_type.title()} Line")
        add_button.clicked.connect(lambda: self.add_line_row(lines_layout))

        layout.addWidget(scroll)
        layout.addWidget(add_button)
        box.lines_layout = lines_layout
        return box

    def add_line_row(self, layout):
        container = QFrame()
        form = QFormLayout()

        label_input = QLineEdit("Placeholder")
        value_input = QLineEdit("2137")

        color_input = QComboBox()
        color_input.addItems(['blue', 'red', 'green', 'cyan', 'purple', 'olive', 'pink', 'gray', 'brown'])

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_input(container))

        form.addRow("Label:", label_input)
        form.addRow("Value:", value_input)
        form.addRow("Color:", color_input)
        form.addRow(remove_button)

        container.setLayout(form)
        layout.addWidget(container)

    @staticmethod
    def collect_lines(layout):
        lines = {}
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget:
                fields = widget.findChildren(QLineEdit)
                combo_box = widget.findChildren(QComboBox)
                if len(fields) >= 2:
                    label = fields[0].text()
                    try:
                        value = float(fields[1].text())
                        color = combo_box[0].currentText()
                        lines[label] = {"place": value, "label": label, "color": color}
                    except ValueError:
                        continue
        return lines

    def add_dataframe(self, file_path, wrapper):
        self.dataframes[file_path] = wrapper
        self.db1_selector.addItem(file_path)
        self.db2_selector.addItem(file_path)
        self.refresh_channel_box("db1")
        self.refresh_channel_box("db2")

    def remove_dataframe(self, file_path):
        if file_path in self.dataframes:
            del self.dataframes[file_path]
        index1 = self.db1_selector.findText(file_path)
        if index1 >= 0:
            self.db1_selector.removeItem(index1)
        index2 = self.db2_selector.findText(file_path)
        if index2 >= 0:
            self.db2_selector.removeItem(index2)

    def refresh_channel_box(self, db_key):
        if db_key == "db1":
            for i in reversed(range(self.db1_box.channel_layout.count())):
                self.db1_box.channel_layout.itemAt(i).widget().deleteLater()
            self.db1_columns.clear()
            self.sync_col1.clear()
            path = self.db1_selector.currentText()
            if path in self.dataframes:
                df = self.dataframes[path].get_dataframe()
                self.sync_col1.addItems(df.columns)
        elif db_key == "db2":
            for i in reversed(range(self.db2_box.channel_layout.count())):
                self.db2_box.channel_layout.itemAt(i).widget().deleteLater()
            self.db2_columns.clear()
            self.sync_col2.clear()
            path = self.db2_selector.currentText()
            if path in self.dataframes:
                df = self.dataframes[path].get_dataframe()
                self.sync_col2.addItems(df.columns)

    @staticmethod
    def remove_input(widget):
        widget.setParent(None)
        widget.deleteLater()

    @staticmethod
    def log(message, log_type):
        match log_type:
            case "DEBUG":
                logger.debug(message)
            case "INFO":
                logger.info(message)
