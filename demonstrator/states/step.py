from PySide6.QtWidgets import (
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QFrame,
    QCheckBox,
    QFileDialog,
    QDialog,
    QListWidget,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator, QIcon
import json
import os
import pandas as pd
import csv
from .style import Styled


def save_all_runs(path, runs):
    copy_runs = list()
    for run in runs:
        copy_run = dict()
        copy_run["timestamp"] = run["timestamp"]
        copy_run["description"] = run["description"]
        copy_run["status"] = run.get("status", None)
        if "dataset" in run:
            copy_run["dataset"] = {
                "module": run["dataset"]["module"],
                "params": run["dataset"]["params"],
            }
        if "model" in run:
            copy_run["model"] = {
                "module": run["model"]["module"],
                "params": run["model"]["params"],
            }
        if "analysis" in run:
            copy_run["analysis"] = {
                "module": run["analysis"]["module"],
                "params": run["analysis"]["params"],
                "return": run["analysis"].get("return", None),
            }
        copy_runs.append(copy_run)
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(copy_runs))


def load_all_runs(path):
    if not os.path.exists(path):
        return list()
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_name(name):
    """Format parameter names for better display."""
    return name.replace("_", " ").capitalize()


class Step(Styled):
    def __init__(self, step_name, stacked_widget, dataset_loaders, runs):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.dataset_loaders = dataset_loaders
        self.first_selection = True  # Track if first selection is made
        self.runs = runs

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.label = QLabel(step_name, self)
        self.label.setStyleSheet("font-size: 30px; font-weight: bold;")
        layout.addWidget(self.label)

        self.dataset_selector = QComboBox(self)
        self.dataset_selector.addItems(
            ["Select a module"] + list(dataset_loaders.keys())
        )
        self.dataset_selector.currentTextChanged.connect(self.update_param_form)
        layout.addWidget(self.dataset_selector)

        # Dataset description section
        self.description_label = QLabel(
            "Select a module to see its description and parameters to fill in.",
            self,
            openExternalLinks=True,
        )
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(
            "font-size: 14px; color: #555; margin-top: 5px;"
        )
        layout.addWidget(self.description_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.param_form = QFormLayout()
        self.param_inputs = {}
        self.form_widget = QWidget()
        self.form_widget.setLayout(self.param_form)
        layout.addWidget(self.form_widget)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #007bff; 
                color: white; 
                border-radius: 5px;
                padding: 6px; 
            }}
            QPushButton:hover {{
                background-color: {self.highlight_color('#007bff')};
            }}
        """
        )
        self.next_button.clicked.connect(self.next)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #dc3545; 
                color: white; 
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.highlight_color('#dc3545')};
            }}
        """
        )
        self.cancel_button.setFixedSize(80, 30)
        self.cancel_button.clicked.connect(self.switch_to_dashboard)

        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.cancel_button)

        layout.addStretch()

        self.description_input = QLineEdit(self)
        self.description_input.setPlaceholderText("Describe your analysis (optional)")
        layout.addWidget(self.description_input)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.defaults = dict()
        self.update_param_form(self.dataset_selector.currentText())

    def update_param_form(self, dataset_name):
        """Update the form based on the selected dataset loader."""
        if self.first_selection and dataset_name != self.dataset_selector.itemText(0):
            self.dataset_selector.removeItem(0)
            self.first_selection = False

        for i in reversed(range(self.param_form.rowCount())):
            self.param_form.removeRow(i)
        self.param_inputs.clear()

        if dataset_name not in self.dataset_loaders:
            self.description_label.setText(
                "Select a dataset loader to see its description."
            )
            return

        loader = self.dataset_loaders[dataset_name]
        self.description_label.setText(
            loader.get(
                "description", f"No description available:<br><b>{dataset_name}</b>"
            )
        )

        self.last_url = None
        self.last_delimiter = None  # never set, placeholder for the future perhaps?
        for name, param_type, default, description in loader["parameters"]:
            default = self.defaults.get(name, default)
            if name == "dataset" or name == "model":
                continue
            param_options = loader.get("parameter_options", {}).get(
                name, []
            )  # Get options if available
            param_widget = self.create_input_widget(
                name, param_type, default, description, param_options
            )
            self.param_form.addRow(param_widget)

    def open_sensitive_modal(self, title, input_field, columns):
        if not isinstance(columns, list):
            path = columns[0].text()
            delimiter = columns[1].text() if columns[1] is not None else None
            if len(path) == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"The previous file was empty and could not be used as reference.",
                )
                return
            if delimiter is not None and len(delimiter) == 0:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"The previous file's delimiter was empty and could not be used as reference</b>",
                )
                return
            try:
                if delimiter is None:
                    try:
                        with open(path, "r") as file:
                            sample = file.read(4096)
                            sniffer = csv.Sniffer()
                            delimiter = sniffer.sniff(sample).delimiter
                            delimiter = str(delimiter)
                    except Exception as e:
                        delimiter = ","
                df = pd.read_csv(
                    path, nrows=3, on_bad_lines="skip", delimiter=delimiter
                )
                columns = df.columns.tolist()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Could not read the previous file to use as reference:<br><b>{str(e)}</b>",
                )
                return

        prev_value = input_field.text()
        prev_selection = set(prev_value.split(","))

        """Open a modal dialog to select sensitive columns."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        layout = QVBoxLayout()

        list_widget = QListWidget(dialog)
        list_widget.addItems(columns)
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(list_widget)

        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.text() in prev_selection:
                item.setSelected(True)

        def cancel():
            input_field.setText(prev_value)
            dialog.accept()

        confirm_button = QPushButton("Cancel", dialog)
        confirm_button.clicked.connect(cancel)
        layout.addWidget(confirm_button)

        confirm_button = QPushButton("Done", dialog)
        confirm_button.clicked.connect(
            lambda: self.set_sensitive_values(dialog, list_widget, input_field)
        )
        layout.addWidget(confirm_button)

        dialog.setLayout(layout)
        dialog.exec()

    def set_sensitive_values(self, dialog, list_widget, input_field):
        """Set selected columns into the input field."""
        selected_items = [item.text() for item in list_widget.selectedItems()]
        input_field.setText(", ".join(selected_items))
        dialog.accept()

    def create_input_widget(
        self, name, param_type, default, description, param_options
    ):
        """Create an appropriate input widget based on the parameter type."""
        param_layout = QHBoxLayout()

        helper = None
        preview = None
        if (
            "numeric" in name
            or "categorical" in name
            or "label" in name
            or "target" in name
            or "ignored" in name
            or "attribute" in name
        ):
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            if self.last_url is not None:
                select_button = QPushButton("...")
                select_button.setToolTip("Select from options")
                select_button.setFixedSize(30, 20)
                select_button.setStyleSheet(
                    "background-color: #dd8; border-radius: 5px;"
                )
                last_url = self.last_url
                last_delimiter = self.last_delimiter
                select_button.clicked.connect(
                    lambda: self.open_sensitive_modal(
                        f"Select {name} columns",
                        input_widget,
                        (last_url, last_delimiter),
                    )
                )
                helper = select_button
        elif name == "sensitive":
            if not self.runs:
                return QWidget()
            columns = self.runs[-1]["dataset"]["return"]
            columns = (
                [""]
                if columns is None or not hasattr(columns, "cols")
                else columns.cols
            )

            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

            select_button = QPushButton("...")
            select_button.setToolTip("Select from options")
            select_button.setFixedSize(30, 20)
            select_button.setStyleSheet("background-color: #dd8; border-radius: 5px;")
            select_button.clicked.connect(
                lambda: self.open_sensitive_modal(
                    "Select sensitive attributes", input_widget, columns
                )
            )

            helper = select_button
        elif param_options:  # If parameter options are provided, use a dropdown
            input_widget = QComboBox(self)
            input_widget.addItems(param_options)
            input_widget.setCurrentText(
                default if default in param_options else param_options[0]
            )
        elif param_type == "int":
            input_widget = QLineEdit(self)
            input_widget.setValidator(QIntValidator())
            input_widget.setText(str(default) if default != "None" else "0")
        elif param_type == "float":
            input_widget = QLineEdit(self)
            input_widget.setValidator(QDoubleValidator())
            input_widget.setText(str(default) if default != "None" else "0.0")
        elif param_type == "bool":
            input_widget = QCheckBox(self)
            input_widget.setChecked(str(default).lower() == "true")
        elif "dir" in name:
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

            file_button = QPushButton("...")
            file_button.setToolTip("Navigate")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet("background-color: #dd8; border-radius: 5px;")
            file_button.clicked.connect(lambda: self.select_dir(input_widget))
            helper = file_button

        elif param_type == "url":
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            self.last_url = input_widget

            file_button = QPushButton("...")
            file_button.setToolTip("Navigate")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet("background-color: #dd8; border-radius: 5px;")
            file_button.clicked.connect(lambda: self.select_path(input_widget))
            helper = file_button

            def preview_file():
                file_path = input_widget.text().strip()
                if not file_path:
                    QMessageBox.warning(self, "Error", "No file selected.")
                    return
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        lines = [file.readline().strip() for _ in range(20)]
                    preview_text = "\n".join(line for line in lines if line)

                    msg_box = QMessageBox(self)
                    msg_box.setWindowTitle("File preview")
                    msg_box.setToolTip("Peek at the first 20 lines")
                    msg_box.setText(preview_text if preview_text else "File is empty.")
                    msg_box.setIcon(
                        QMessageBox.Icon.NoIcon
                    )  # Removes the information icon
                    msg_box.exec_()

                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Could not read the file or failed to convert it to a human-friendly format:\n{str(e)}",
                    )

            file_button = QPushButton("Preview")
            file_button.setFixedSize(50, 20)
            file_button.setStyleSheet("background-color: #dbd; border-radius: 5px;")
            file_button.clicked.connect(preview_file)
            preview = file_button

        elif name == "delimiter":
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            last_url = self.last_url

            def recommend_delimiter():
                path = last_url.text()
                if len(path) == 0:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"The previous file was empty and could not be used as reference.",
                    )
                    return
                try:
                    with open(path, "r") as file:
                        sample = file.read(4096)
                        sniffer = csv.Sniffer()
                        delimiter = sniffer.sniff(sample).delimiter
                        delimiter = str(delimiter)
                        input_widget.setText(delimiter)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Could not read the previous file to use as reference:<br><b>{str(e)}</b>",
                    )

            file_button = QPushButton("Find")
            file_button.setToolTip("Autodetect based on csv rules")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet("background-color: #dd8; border-radius: 5px;")
            file_button.clicked.connect(recommend_delimiter)
            preview = file_button

        else:  # Default to a normal text field
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

        self.param_inputs[name] = input_widget  # Store reference to input field

        label = QLabel(format_name(name))
        label.setFixedSize(150, 20)
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        help_button = QPushButton("?")
        help_button.setFixedSize(30, 20)
        help_button.setStyleSheet(
            "background-color: #ddd; border-radius: 10px; font-weight: bold;"
        )
        help_button.setToolTip("Parameter info")
        help_button.clicked.connect(
            lambda: self.show_help_popup(format_name(name), description)
        )

        param_layout.addWidget(label)
        param_layout.addWidget(help_button)
        if helper is not None:
            param_layout.addWidget(helper)
        if preview is not None:
            param_layout.addWidget(preview)
        param_layout.addWidget(input_widget)

        param_widget = QWidget()
        param_widget.setLayout(param_layout)
        return param_widget

    def select_dir(self, input_field):
        path = QFileDialog.getExistingDirectory(self, "Select directory")
        if path:
            input_field.setText(path)

    def select_path(self, input_field):
        path = QFileDialog.getOpenFileName(self, "Select file")
        if path:
            input_field.setText(path[0])

    def show_help_popup(self, param_name, description):
        """Show a popup window with the parameter description."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Parameter info")
        msg.setText(description)
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.exec()

    def save(self, step):
        pipeline = self.runs[-1]
        dataset_name = self.dataset_selector.currentText()
        params = {}
        for param, field in self.param_inputs.items():
            if isinstance(field, QCheckBox):
                params[param] = field.isChecked()
            elif isinstance(field, QComboBox):
                params[param] = field.currentText()
            else:
                params[param] = field.text()
        pipeline[step] = {"module": dataset_name, "params": params}
        pipeline["description"] = self.description_input.text().strip()

    def show_error_message(self, message):
        error_msg = QMessageBox(self)
        if message[0] == "'" and message[-1] == "'":
            message = message[1:-1]
        message = "The following issue must be addressed:<br><b>" + message + "</b>"
        error_msg.setWindowTitle("Error")
        error_msg.setText(message)
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setModal(True)
        error_msg.exec()
