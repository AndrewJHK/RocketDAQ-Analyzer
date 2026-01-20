from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QStackedWidget, QLabel,
)
from gui.panels.data_acquisition_panel import UploadPanel
from gui.panels.data_processing_panel import DataProcessingPanel
from gui.panels.flight_plot_panel import FlightPlotPanel
from gui.panels.plotting_panel import PlottingPanel
from src.data_processing import DataFrameWrapper
import os


class FileWidget(QWidget):
    def __init__(self, file_path, delete_callback):
        super().__init__()

        self.file_path = file_path
        self.delete_callback = delete_callback

        layout = QHBoxLayout()
        self.label = QLabel(os.path.basename(file_path))
        self.delete_button = QPushButton("Delete")
        self.delete_button.setFixedWidth(70)
        self.delete_button.clicked.connect(self.handle_delete)

        layout.addWidget(self.label)
        layout.addWidget(self.delete_button)
        self.setLayout(layout)

    def handle_delete(self):
        self.delete_callback(self)


class PostProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Postprocessing GUI")

        self.file_widgets = []
        self.dataframes = {}
        self.active_button = None

        main_layout = QHBoxLayout()

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(10)

        self.load_button = QPushButton("1. Load data")
        self.flight_plot_button = QPushButton("2. Flight plot")
        self.processing_button = QPushButton("3. Data processing")
        self.plotter_button = QPushButton("4. Plotting")

        self.buttons = [
            self.load_button,
            self.flight_plot_button,
            self.processing_button,
            self.plotter_button
        ]

        for btn in self.buttons:
            btn.setCheckable(True)
            btn.setStyleSheet("QPushButton { padding: 6px; } QPushButton:checked { background-color: lightblue; }")

        self.load_button.clicked.connect(lambda: self.change_panel(0, self.load_button))
        self.flight_plot_button.clicked.connect(lambda: self.change_panel(1, self.flight_plot_button))
        self.processing_button.clicked.connect(lambda: self.change_panel(2, self.processing_button))
        self.plotter_button.clicked.connect(lambda: self.change_panel(3, self.plotter_button))

        sidebar_layout.addWidget(self.load_button)
        sidebar_layout.addWidget(self.flight_plot_button)
        sidebar_layout.addWidget(self.processing_button)
        sidebar_layout.addWidget(self.plotter_button)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.file_list_widget = QWidget()
        self.file_list_layout = QVBoxLayout()
        self.file_list_widget.setLayout(self.file_list_layout)
        self.scroll_area.setWidget(self.file_list_widget)
        sidebar_layout.addWidget(self.scroll_area, stretch=1)

        self.stack = QStackedWidget()
        self.upload_panel = UploadPanel(add_callback=self.add_dataframe, add_file_widget=self.add_file)
        self.flight_panel = FlightPlotPanel()
        self.processing_panel = DataProcessingPanel()
        self.plotting_panel = PlottingPanel()

        self.stack.addWidget(self.upload_panel)
        self.stack.addWidget(self.flight_panel)
        self.stack.addWidget(self.processing_panel)
        self.stack.addWidget(self.plotting_panel)

        main_layout.addLayout(sidebar_layout, stretch=1)
        main_layout.addWidget(self.stack, stretch=4)

        self.setLayout(main_layout)
        self.change_panel(0, self.load_button)

    def change_panel(self, index, button):
        self.stack.setCurrentIndex(index)
        if self.active_button:
            self.active_button.setChecked(False)
        button.setChecked(True)
        self.active_button = button
        if index == 2:
            self.processing_panel.update_columns()

    def add_dataframe(self, file_path, wrapper: DataFrameWrapper):
        self.dataframes[file_path] = wrapper
        self.flight_panel.add_dataframe(file_path, wrapper)
        self.processing_panel.add_dataframe(file_path, wrapper)
        self.plotting_panel.add_dataframe(file_path, wrapper)

    def _is_duplicate(self, file_path):
        return any(w.file_path == file_path for w in self.file_widgets)

    def add_file(self, file_path):
        if self._is_duplicate(file_path):
            return
        file_widget = FileWidget(file_path, self.remove_file)
        self.file_widgets.append(file_widget)
        self.file_list_layout.addWidget(file_widget)

    def remove_file(self, widget):
        file_path = widget.file_path
        self.file_widgets.remove(widget)
        widget.setParent(None)
        widget.deleteLater()
        if file_path in self.dataframes:
            del self.dataframes[file_path]

        self.upload_panel.remove_dataframe(file_path)
        self.flight_panel.remove_dataframe(file_path)
        self.processing_panel.remove_dataframe(file_path)
        self.plotting_panel.remove_dataframe(file_path)
