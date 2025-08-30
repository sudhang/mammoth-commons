from PySide6.QtWidgets import (
    QPushButton,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QScrollArea,
    QMessageBox,
    QLineEdit,
)
from PySide6.QtCore import Qt
from datetime import datetime
from .step import save_all_runs
from .style import Styled
import re
from functools import partial


def now():
    return datetime.now().strftime("%y-%m-%d %H:%M")


class Dashboard(Styled):
    def __init__(self, stacked_widget, runs, tag_descriptions):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.label = QLabel("Fairness analysis", self)
        self.label.setStyleSheet("font-size: 30px; font-weight: bold;")
        self.main_layout.addWidget(self.label)

        new_button = self.create_icon_button(
            "➕", "#007bff", "New analysis", self.create_new_item
        )
        new_button.setFixedSize(40, 40)

        search_field = QLineEdit(self)
        search_field.setPlaceholderText("Search...")
        search_field.setFixedWidth(200)
        search_field.textChanged.connect(
            self.filter_runs
        )  # Connect to filtering method

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(new_button)
        button_layout.addWidget(search_field)
        self.main_layout.addLayout(button_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Content Widget
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(0)
        self.scroll_area.setWidget(self.content_widget)

        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)
        self.tag_descriptions = tag_descriptions

        self.invisible_runs = set()
        self.refresh_dashboard()

    def filter_runs(self, text):
        prev = self.invisible_runs
        self.invisible_runs = set()
        for index, run in enumerate(self.runs):
            if text.lower() in run["description"].lower():
                continue
            if text.lower() in run.get("dataset", dict()).get("module", "").lower():
                continue
            if text.lower() in run.get("model", dict()).get("module", "").lower():
                continue
            if text.lower() in run.get("analysis", dict()).get("module", "").lower():
                continue
            if text.lower() in format_run(run).lower():
                continue
            self.invisible_runs.add(index)
        # refresh but only if something changed
        if (
            len(prev - self.invisible_runs) == 0
            and len(self.invisible_runs - prev) == 0
        ):
            return
        self.refresh_dashboard()

    def view_result(self, index):
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.slideToWidget(4)

    def edit_item(self, index):
        if self.runs[index].get("status", "") != "completed":
            reply = QMessageBox.StandardButton.Yes
        else:
            reply = QMessageBox.question(
                self,
                "Edit?",
                f"You can change modules and modify parameters of {format_run(self.runs[index])}. "
                "However, this will also remove its results. Consider creating a variation if you want to preserve current results.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.runs[index]["timestamp"] = now()
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.slideToWidget(1)

    def create_variation(self, index):
        new_run = self.runs[index].copy()
        new_run["status"] = "new"
        new_run["timestamp"] = now()
        self.runs.append(new_run)
        self.stacked_widget.slideToWidget(1)

    def create_new_item(self):
        self.runs.append(
            {"description": "", "timestamp": now(), "status": "in_progress"}
        )
        self.stacked_widget.slideToWidget(1)
        self.refresh_dashboard()

    def delete_item(self, index):
        reply = QMessageBox.question(
            self,
            "Delete?",
            f"Confirm the deletion of {format_run(self.runs[index])}.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.runs.pop(index)
        self.refresh_dashboard()
        save_all_runs("history.json", self.runs)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def showEvent(self, event):
        self.refresh_dashboard()

    def refresh_dashboard(self):
        self.clear_layout(self.layout)
        visual_pos = -1
        sorted_items = list(
            sorted(
                enumerate(self.runs),
                key=lambda x: x[1]["description"] + x[1]["status"] + x[1]["timestamp"],
            )
        )
        sorted_items = [
            (index, run)
            for index, run in sorted_items
            if index not in self.invisible_runs
        ]

        prev_has_same_next_tags = False
        for index, run in sorted_items:
            visual_pos += 1

            tags = []
            if "dataset" in run:
                tags.append(run["dataset"]["module"])
            if "model" in run:
                tags.append(run["model"]["module"])
            if "analysis" in run:
                tags.append(run["analysis"]["module"])
            has_same_next_tags = False
            if visual_pos < len(sorted_items) - 1 and run["status"] == "completed":
                next_tags = []
                next_run = sorted_items[visual_pos + 1][1]
                if "dataset" in next_run:
                    next_tags.append(next_run["dataset"]["module"])
                if "model" in next_run:
                    next_tags.append(next_run["model"]["module"])
                if "analysis" in next_run:
                    next_tags.append(next_run["analysis"]["module"])
                has_same_next_tags = (
                    next_run["status"] == run["status"]
                    and next_run["description"] == run["description"]
                    and len(set(tags) - set(next_tags)) == 0
                    and len(set(next_tags) - set(tags)) == 0
                )

            button_color = (
                ("#bb8888" if "fail" in format_run(run) else "#88bb88")
                if run["status"] == "completed"
                else "#d69e02"
            )
            run_button = QPushButton(self)
            run_button.setFixedHeight(90)
            button_label = QLabel(
                format_run(run, simpler=has_same_next_tags or prev_has_same_next_tags),
                run_button,
            )
            button_label.setTextFormat(Qt.TextFormat.RichText)
            button_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            button_label.setWordWrap(True)
            button_layout = QVBoxLayout(run_button)
            button_layout.addWidget(button_label)
            button_layout.setContentsMargins(5, 5, 5, 5)
            run_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {button_color};
                    color: black;
                    border-radius: 5px;
                    font-size: 16px;
                    border: 1px solid black;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    border: 2px solid black;
                    background-color: {self.highlight_color(button_color)};
                }}
                QPushButton:pressed {{
                    background-color: {self.highlight_color(self.highlight_color(button_color))};
                }}
            """
            )
            run_button.clicked.connect(
                partial(
                    lambda i=index, r=run: (
                        self.view_result(i)
                        if r["status"] == "completed"
                        else self.edit_item(i)
                    )
                )
            )

            # Stack button and tags in a vertical layout
            button_with_tags_layout = QVBoxLayout()

            if not prev_has_same_next_tags and has_same_next_tags:
                button_with_tags_layout = QVBoxLayout()
                label = QLabel(run["description"], self)
                label.setStyleSheet("font-size: 26px; font-weight: bold;")
                label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                button_with_tags_layout.addWidget(label)

            button_with_tags_layout.addWidget(run_button)
            if not has_same_next_tags:
                # Create a container for tags
                tag_container = QHBoxLayout()
                tag_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                tag_container.setContentsMargins(
                    0, -30, 0, 30
                )  # Slight overlap with button, space below
                for tag in tags:
                    tag_container.addWidget(
                        self.create_tag_button(
                            f" {tag} ",
                            "Module info",
                            partial(lambda t=tag: self.show_tag_description(t)),
                        )
                    )
                if run["status"] == "completed":
                    tag_container.addWidget(
                        self.create_icon_button(
                            "➕",
                            "#007bff",
                            "New variation",
                            partial(lambda i=index: self.create_variation(i)),
                        )
                    )
                tag_container.addWidget(
                    self.create_icon_button(
                        "🗑",
                        "#dc3545",
                        "Delete",
                        partial(lambda i=index: self.delete_item(i)),
                    )
                )
                button_with_tags_layout.addLayout(tag_container)

            if has_same_next_tags or prev_has_same_next_tags:
                run_button.setContentsMargins(0, 0, 0, 0)
                run_button.setFixedHeight(60)

            # button_with_tags_layout.setSpacing(-5)  # Reduce spacing for overlap effect
            button_with_tags_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            # Main row layout with full width
            row_layout = QHBoxLayout()
            row_layout.addLayout(button_with_tags_layout)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)

            self.layout.addLayout(row_layout)
            prev_has_same_next_tags = has_same_next_tags

        self.content_widget.adjustSize()

    def show_tag_description(self, tag):
        """Show description of a tag."""
        msg = QMessageBox()
        msg.setWindowTitle("Module info")
        msg.setText(self.tag_descriptions.get(tag, "No description available."))
        msg.exec()


def format_run(run, simpler=False):
    # this function is a mess because it's easier to try things out this way
    try:
        match = re.search(
            r"<h1\b[^>]*>.*?</h1>",
            run.get("analysis", dict()).get("return", ""),
            re.DOTALL,
        )
        if match:
            match = (
                "" if simpler or len(run["description"]) == 0 else ": "
            ) + match.group().replace("h1", "span")
        else:
            match = "✎"
    except Exception:
        match = "✎"
    if simpler:
        return f'<h2 style="margin: 0px;">{match}</h2>Created at {run["timestamp"]}'
    return f'<h1 style="margin: 0px;">{"" if simpler else run["description"]}{match}</h1>Created at {run["timestamp"]}'
