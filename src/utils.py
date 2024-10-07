import json
import librosa
import numpy as np
import os
import torch
from . import Models


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


def load_subset_model(subset: str = "all", target_list: list = None):
    if subset != "dimensions":

        best_model_folder = (
            "data/models/regression_models/ML_including_industry_feature/multi_dimension_"
            + subset
            + "/version_best/checkpoints"
        )
        model_files = [
            file for file in os.listdir(best_model_folder) if file.endswith(".ckpt")
        ]
        # load model
        model = Models.DNNRegressor.load_from_checkpoint(
            best_model_folder + "/" + model_files[0]
        )
        return model
    else:
        # load all models for all dimensions
        models = {}
        for target in target_list:
            best_model_folder = (
                "data/models/regression_models/ML_including_industry_feature/single_dimension/"
                + target
                + "/version_best/checkpoints"
            )
            model_files = [
                file for file in os.listdir(best_model_folder) if file.endswith(".ckpt")
            ]
            # load model
            model = Models.DNNRegressor.load_from_checkpoint(
                best_model_folder + "/" + model_files[0]
            )
            models[target] = model
        return models


def predict(model, input_features):

    assert model.hparams["input_dim"] == input_features.shape[0]
    # convert waveform to input features
    input_features = torch.tensor(input_features, dtype=torch.float32)
    input_features = input_features.clone().detach().requires_grad_(True)
    input_features = input_features.to(model.device)
    model.eval()
    prediction = model(input_features)

    return prediction.detach().cpu().numpy()
