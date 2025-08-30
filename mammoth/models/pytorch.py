from mammoth.models.predictor import Predictor
from mammoth.datasets.image_pairs import ImagePairs


def get_predictions(threshold, embed1, embed2):
    import torch

    embed1 = torch.nn.functional.normalize(embed1, p=2, dim=1)
    embed2 = torch.nn.functional.normalize(embed2, p=2, dim=1)
    diff = embed1 - embed2
    dist = torch.sum(diff**2, dim=1)
    predict_issame = (dist < threshold).int()
    return predict_issame


class Pytorch(Predictor):
    def __init__(self, model):
        self.model = model

    def predict(self, dataset, sensitive):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        dataloader = dataset.to_torch(sensitive)

        model.eval()
        all_predictions = []
        all_labels = []
        all_sensitive = [[] for _ in sensitive]
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(dataset, ImagePairs):
                    input1 = batch[0].to(device)
                    input2 = batch[1].to(device)
                    targets = batch[2]
                    sens = batch[3]
                    output1 = model(input1)
                    output2 = model(input2)
                    predictions = get_predictions(1.3, output1, output2)
                else:
                    inputs = batch[0].to(device)
                    targets = batch[1]
                    sens = batch[2]
                    outputs = model(inputs)
                    predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_labels.append(targets.cpu())
                for i in range(len(sensitive)):
                    all_sensitive[i] += [sens[i].cpu() for i in range(len(sens))]
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        dataset.labels = {"0": 1 - all_labels, "1": all_labels}
        dataset.data = {
            name: torch.cat(value) for name, value in zip(sensitive, all_sensitive)
        }
        return all_predictions
