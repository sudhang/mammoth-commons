from mammoth.models.predictor import Predictor
from mammoth.datasets.image_pairs import ImagePairs
import numpy as np
import onnxruntime as ort


def get_predictions(threshold, embed1, embed2):
    embed1 = embed1 / np.linalg.norm(embed1, axis=1, keepdims=True)  # L2 normalization
    embed2 = embed2 / np.linalg.norm(embed2, axis=1, keepdims=True)  # L2 normalization
    diff = embed1 - embed2
    dist = np.sum(diff**2, axis=1)
    predict_issame = (dist < threshold).astype(int)
    return predict_issame


class ONNX(Predictor):
    def __init__(self, model):
        self.model = model
        # Initialize the ONNX runtime session
        self.ort_session = ort.InferenceSession(model)

    def predict(self, dataset, sensitive):
        all_predictions = []
        all_labels = []
        all_sensitive = [[] for _ in sensitive]
        dataloader = dataset.to_numpy(sensitive)
        for batch in dataloader:
            if isinstance(dataset, ImagePairs):
                input1 = batch[0]  # Assuming batch[0] is a NumPy array
                input2 = batch[1]
                targets = batch[2]
                sens = batch[3]

                # Run inference for both inputs
                output1 = self.ort_session.run(
                    None, {self.ort_session.get_inputs()[0].name: input1}
                )
                output2 = self.ort_session.run(
                    None, {self.ort_session.get_inputs()[0].name: input2}
                )

                predictions = get_predictions(1.3, output1[0], output2[0])
            else:
                inputs = batch[0]
                targets = batch[1]
                sens = batch[2]

                # Run inference
                outputs = self.ort_session.run(
                    None, {self.ort_session.get_inputs()[0].name: inputs}
                )
                predictions = np.argmax(outputs[0], axis=1)

            all_predictions.append(predictions)
            all_labels.append(targets)
            for i in range(len(sensitive)):
                all_sensitive[i] += [sens[i] for i in range(len(sens))]

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        dataset.labels = {"0": 1 - all_labels, "1": all_labels}
        dataset.data = {
            name: np.concatenate(value) for name, value in zip(sensitive, all_sensitive)
        }
        return all_predictions
