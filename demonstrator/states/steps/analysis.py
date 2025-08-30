from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QThread, Signal, QMutex
from demonstrator.backend.loaders import registry
from demonstrator.states.step import Step, save_all_runs
import traceback


class AnalysisThread(QThread):
    finished_success = Signal(object)
    finished_failure = Signal(str)
    canceled = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._is_canceled = False
        self.mutex = QMutex()

    def run(self):
        try:
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            args = self.pipeline["analysis"]["params"]
            dataset = self.pipeline["dataset"]["return"]
            model = self.pipeline["model"]["return"]
            sensitive = args.get("sensitive", "")
            if "," in sensitive:
                sensitive = sensitive.split(",")
            elif sensitive == "":
                sensitive = []
            else:
                sensitive = [sensitive]
            sensitive = [s.strip() for s in sensitive]
            args = {
                k: v
                for k, v in args.items()
                if k not in ["dataset", "model", "sensitive"]
            }
            self.pipeline["analysis"]["return"] = registry.name_to_runnable[
                self.pipeline["analysis"]["module"]
            ](dataset, model, sensitive, **args).text()
            self.pipeline["dataset"]["return"] = None
            self.pipeline["model"]["return"] = None
            self.pipeline["status"] = "completed"

            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()

            self.finished_success.emit(self.pipeline)
        except Exception as e:
            traceback.print_exception(e)
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()


class SelectAnalysis(Step):
    def showEvent(self, event):
        pipeline = self.runs[-1]
        self.description_input.setText(pipeline["description"])
        compatible_methods = [
            method
            for method, entries in registry.analysis_methods.items()
            if issubclass(
                registry.parameters_to_class[pipeline["dataset"]["module"]]["return"],
                registry.parameters_to_class[method][entries["parameters"][0][0]],
            )
            and issubclass(
                registry.parameters_to_class[pipeline["model"]["module"]]["return"],
                registry.parameters_to_class[method][entries["parameters"][1][0]],
            )
        ]
        self.dataset_selector.clear()
        self.dataset_selector.addItems(
            ["Select a fairness analysis method"] + compatible_methods
        )
        self.defaults = self.runs[-1].get("analysis", dict()).get("params", dict())
        # self.update_param_form(self.runs[-1].get("analysis", dict()).get("module", "Select a fairness analysis method"))
        self.dataset_selector.setCurrentIndex(
            self.dataset_selector.findText(
                self.runs[-1]
                .get("analysis", dict())
                .get("module", "Select a fairness analysis method")
            )
        )
        super().showEvent(event)

    def next(self):
        self.save("analysis")
        pipeline = self.runs[-1]

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Running fairness analysis")
        self.loading_message.setText(
            "Please wait while the fairness analysis is running..."
        )
        self.loading_message.setStandardButtons(QMessageBox.StandardButton.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.StandardButton.Cancel).clicked.connect(
            self.cancel_loading
        )
        self.loading_message.show()

        # Start the analysis thread
        self.thread = AnalysisThread(pipeline)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.slideToWidget(4)
        save_all_runs("history.json", self.runs)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)
        save_all_runs("history.json", self.runs)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()  # Ensure thread is properly finished
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("analysis")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.slideToWidget(0)
        save_all_runs("history.json", self.runs)

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        event.accept()
