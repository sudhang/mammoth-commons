import numpy as np
from mammoth.models.predictor import Predictor


class ONNX(Predictor):
    def __init__(self, model_bytes, np_type=np.float64):
        self.model_bytes = model_bytes
        self.np_type = np_type

    def predict(self, dataset, sensitive):
        x = (
            dataset
            if isinstance(dataset, np.ndarray)
            else dataset.to_features(sensitive)
        )
        import onnxruntime as rt
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

        sess = rt.InferenceSession(self.model_bytes, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        try:
            return sess.run([label_name], {input_name: x.astype(self.np_type)})[0]
        except InvalidArgument as e:
            raise Exception(
                "The ONNx loader's runtime encountered an error that typically occurs "
                "when the selected dataset is incompatible to the loaded model. "
                "Consult with the model provider whether your are loading the "
                "model properly. If you are investigating a dataset, "
                "consider switching to trained-on-the-fly model loaders.<br><br>"
                '<details><summary class="btn btn-secondary">Details</summary><br><br>'
                "<pre>" + str(e) + "</pre></details>"
            )
