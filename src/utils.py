import json
import os

import joblib
import librosa
import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor

from .Models import InferenceDNNRegressor


def load_global_variables() -> dict:
    """
    Load global variables from a json file.
    """

    dict_path = "./global_variables.json"
    with open(dict_path, "r") as file:
        global_variables = json.load(file)
    return global_variables


def load_waveforms(file_names: list, sample_rate: int = 22050) -> list:
    """
    Load audio files and return waveforms
    """
    waveforms = []
    for file_name in file_names:
        waveform, _ = librosa.load(file_name, sr=sample_rate)
        waveforms.append(waveform)
    print(f"{len(waveforms)} waveforms loaded")
    return waveforms


def predict(model, input_features):
    # check if model is of type DNNRegressor
    if isinstance(model, InferenceDNNRegressor):
        # convert waveform to input features
        input_features = torch.tensor(input_features, dtype=torch.float32)
        input_features = input_features.clone().detach().requires_grad_(True)
        input_features = input_features.to(model.device)
        model.eval()
        prediction = model(input_features)

        return prediction.detach().cpu().numpy()

    # check if model is of type RandomForestRegressor
    elif isinstance(model, RandomForestRegressor):
        prediction = model.predict(input_features.reshape(1, -1))
        # return prediction as 1D array
        return prediction.reshape(-1)


def load_multioutput_model(
    subset: str = "all", include_industry: bool = True, return_model_path: bool = False
):
    if subset == "all":
        if include_industry:
            best_model_file = "./models/regression_models/multi_dimension_all/industry/241016_model_with_insdustry.pkl"
        else:
            best_model_file = "./models/regression_models/multi_dimension_all/no_industry/241016_model_no_industry.pkl"
        model = joblib.load(best_model_file)

    if subset == "functional":
        if include_industry:
            best_model_file = "./models/regression_models/multidimension_f_bi_level/industry/functional/241022_model.pkl"
        else:
            best_model_file = "./models/regression_models/multidimension_f_bi_level/no_industry/functional/241022_model.pkl"

        model = joblib.load(best_model_file)

    if subset == "brand_identity":
        if include_industry:
            best_model_folder = "./models/regression_models/multidimension_f_bi_level/industry/brand_identity/lightning_logs/241022_version_best/"
            best_model_checkpoint = (
                best_model_folder + "checkpoints/epoch=59-step=1500.ckpt"
            )
            model = InferenceDNNRegressor(industry=True)
        else:
            best_model_folder = "./models/regression_models/multidimension_f_bi_level/no_industry/brand_identity/lightning_logs/241022_version_best/"
            best_model_checkpoint = (
                best_model_folder + "checkpoints/epoch=59-step=2040.ckpt"
            )
            model = InferenceDNNRegressor(industry=False)

        model.load_state_dict(torch.load(best_model_checkpoint)["state_dict"])

    # when saving the prediction in a dictionary the model path is stored
    if return_model_path:
        return model, best_model_file
    else:
        return model
