# imports
import os
import argparse
import numpy as np
import pandas as pd
import librosa

from src.utils import (
    load_global_variables,
    load_multioutput_model,
    predict,
)
from src.input_pipeline.input_features import (
    get_model_input,
    append_industry_to_model_input,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sound_file", type=str, default="data/audios/AMG_Mobility-003.mp3"
)
parser.add_argument("--industry", type=str, default=None)
parser.add_argument("--output_folder", type=str, default="data/predictions")
parser.add_argument(
    "--subset",
    type=str,
    default="all",
    help="Subset of the model to use. Options: all, functional, brand_identity",
)
args = parser.parse_args()

assert args.subset in [
    "all",
    "functional",
    "brand_identity",
], "Invalid subset argument. The options are: all, functional, brand_identity"

# load resources and models
global_variables = load_global_variables()
if args.industry is None:
    model, model_path = load_multioutput_model(
        subset=args.subset, include_industry=False, return_model_path=True
    )
else:
    model, model_path = load_multioutput_model(
        subset=args.subset, return_model_path=True
    )
print(f"Predicting with model: {model_path}")

# define target list depending on subset
if args.subset == "all":
    target_list = global_variables["target_list"]
elif args.subset == "functional":
    target_list = global_variables["status_list"] + global_variables["appeal_list"]
elif args.subset == "brand_identity":
    target_list = global_variables["brand_identity_list"]


# load waveform and create input features
print(f"Predicting sound file: {args.sound_file}")
waveform, _ = librosa.load(args.sound_file, sr=global_variables["sample_rate"])
input_features = get_model_input(waveform, global_variables)

if args.industry is None:
    pass
else:
    input_features = append_industry_to_model_input(
        input_features, args.industry, global_variables
    )

# predict and write to dict
prediction = predict(model, input_features)

model_prediction_dict = {}
# to track which model type and version was used
model_prediction_dict["model"] = model_path
prediction_dict = {}
for i, target in enumerate(target_list):
    prediction_dict[target] = prediction[i]
model_prediction_dict["predictions"] = prediction_dict

# save in folder
output_path = os.path.join(
    args.output_folder, os.path.basename(args.sound_file) + ".json"
)
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
pd.Series(model_prediction_dict).to_json(output_path, orient="index", indent=4)

print(f"Prediction saved to: {output_path}")
