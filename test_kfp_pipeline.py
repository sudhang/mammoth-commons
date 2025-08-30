from catalogue.model_loaders.onnx import model_onnx
from catalogue.dataset_loaders.auto_csv import data_auto_csv
from catalogue.metrics.model_card import model_card
from typing import Dict, List
from kfp import dsl, local


def test_kfp_pipeline():

    @dsl.pipeline(name="pipeline_test")
    def pipeline(
        model_onnx__params: Dict,
        data_auto_csv__params: Dict,
        sensitive: List,
        model_card__params: Dict,
    ):
        model_onnx_task = model_onnx(model_onnx__params=model_onnx__params)
        data_auto_csv_task = data_auto_csv(data_auto_csv__params=data_auto_csv__params)
        model_card_task = model_card(
            model_card__params=model_card__params,
            sensitive=sensitive,
            dataset=data_auto_csv_task.outputs["output"],
            model=model_onnx_task.outputs["output"],
        )

    # IMPORTANT: Components need to be build to have the docker image before testing with this file
    # Build components commands
    # Run the following commands to build the relevant components

    # kfp component build . --component-filepattern catalogue/model_loaders/onnx.py
    # kfp component build . --component-filepattern catalogue/dataset_loaders/auto_csv.py
    # kfp component build . --component-filepattern catalogue/metrics/model_card.py

    # Test pipeline execution
    local.init(runner=local.DockerRunner())

    pipeline(
        model_onnx__params={
            "path": "https://github.com/mammoth-eu/mammoth-commons/raw/refs/heads/dev/data/model.onnx"
        },
        data_auto_csv__params={
            "path": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip/bank/bank.csv"
        },
        sensitive=["marital"],
        model_card__params={"compare_groups": "Pairwise", "intersectional": False},
    )


test_kfp_pipeline()
