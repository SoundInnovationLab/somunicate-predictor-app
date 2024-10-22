import streamlit as st
import numpy as np
import pandas as pd
import librosa

from src.utils import load_global_variables, load_multioutput_model, predict
from src.input_pipeline.input_features import (
    get_model_input,
    append_industry_to_model_input,
)

st.set_page_config(layout="wide")

global_variables = load_global_variables()

st.title("Predict the Status, Appeal and Brand Identity of UX sounds")


# option for selecting the type of model will be included soon
model_type = st.radio(
    "Select the type of model",
    [
        "Multi-Output Model (overall)",
        "Multi-Output Model (Functional & Brand Identity separately)",
    ],
    horizontal=True,
)

files = st.file_uploader(
    "Upload your functional sound(s)", type=["mp3", "wav"], accept_multiple_files=True
)
include_industry = st.toggle(
    "Use industry as input",
    value=True,
    help="The model can predict the percieved expression of the UX sound better, if information on the industry is provided. If it is not provided, the prediction still works.",
)

if include_industry:
    model_all = load_multioutput_model()
    model_functional = load_multioutput_model(subset="functional")
    model_brand_identity = load_multioutput_model(subset="brand_identity")
else:
    model_all = load_multioutput_model(include_industry=False)
    model_functional = load_multioutput_model(
        subset="functional", include_industry=False
    )
    model_brand_identity = load_multioutput_model(
        subset="brand_identity", include_industry=False
    )

if files:
    # somehow the load_waveforms function is not working with the list of UploadedFile objects
    for f_idx, file in enumerate(files):

        waveform, _ = librosa.load(file, sr=global_variables["sample_rate"])

        with st.expander(f"Sound {f_idx + 1}: {file.name}"):

            col1, col2 = st.columns(2)
            col1.audio(file, format="audio/wav")
            # calculate input features here so they don't get
            # recalculated when changing to industry model or changing the industry
            input_features = get_model_input(waveform, global_variables)
            with col2:
                if include_industry:
                    industry_input = st.radio(
                        "Select industry",
                        global_variables["industry_list"],
                        horizontal=True,
                        key=f"industry_{f_idx}",
                    )
                    # get input feature array with industry (one-hot encoeded)
                    input_features = append_industry_to_model_input(
                        input_features, industry_input, global_variables
                    )

            if model_type == "Multi-Output Model (overall)":
                prediction = predict(model_all, input_features)
                pred_df = pd.DataFrame(
                    {
                        "Dimension": global_variables["target_list"],
                        "Prediction": prediction,
                    }
                )

            elif (
                model_type
                == "Multi-Output Model (Functional & Brand Identity separately)"
            ):
                functional = predict(model_functional, input_features)
                brand_identity = predict(model_brand_identity, input_features)
                pred_df = pd.DataFrame(
                    {
                        "Dimension": global_variables["status_list"]
                        + global_variables["appeal_list"]
                        + global_variables["brand_identity_list"],
                        "Prediction": np.concatenate([functional, brand_identity]),
                    }
                )

            col_status, col_appeal, col_brand = st.columns(3)
            with col_status:
                st.header("Status Dimension")
                st.dataframe(
                    pred_df[
                        pred_df["Dimension"].isin(global_variables["status_list"])
                    ].sort_values(by="Prediction", ascending=False)
                )
            with col_appeal:
                st.header("Appeal Dimension")
                st.dataframe(
                    pred_df[
                        pred_df["Dimension"].isin(global_variables["appeal_list"])
                    ].sort_values(by="Prediction", ascending=False)
                )
            with col_brand:
                st.header("Brand Identity Dimension")
                st.dataframe(
                    pred_df[
                        pred_df["Dimension"].isin(
                            global_variables["brand_identity_list"]
                        )
                    ].sort_values(by="Prediction", ascending=False)
                )
