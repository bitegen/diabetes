import typing as t

import pandas as pd

from diabetes_model import __version__ as _version
from diabetes_model.config.core import config
from diabetes_model.processing.data_manager import load_pipeline

pipeline_file_name = f"{config.app_config.pipeline_save_file}{config.app_config.version}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    # validated_data, errors = validate_inputs(input_data=data)

    results: t.Dict[str, t.Any] = {"preds": None, "probs": None, "version": config.app_config.version}

    preds = _price_pipe.predict(X=input_data[config.model_config_params.features])
    probs = _price_pipe.predict_proba(X=input_data[config.model_config_params.features])[:, 1]

    # Fill the results dict
    results["preds"] = [pred for pred in preds]
    results["probs"] = [prob for prob in probs]

    return results
