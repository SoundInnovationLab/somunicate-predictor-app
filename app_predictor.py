import streamlit as st
import librosa
from src.utils import load_global_variables, load_subset_model
from src.input_pipeline.input_features import get_model_input

global_variables = load_global_variables()

st.title("Predict the Status, Appeal and Brand Identity of UX sounds")

files = st.file_uploader(
    "Upload your functional sound(s)", type=["mp3", "wav"], accept_multiple_files=True
)

if files:

    # somehow the load_waveforms function is not working with the list of UploadedFile objects
    for file in files:
        waveform, _ = librosa.load(file, sr=global_variables["sample_rate"])

        col1, col2 = st.columns(2)
        col1.audio(file, format="audio/wav")
        with col2:
            industry_input = st.radio(
                "Select industry",
                global_variables["industry_list"],
            )
        # industry rating is part of the model input features

        input_features = get_model_input(waveform, industry_input, global_variables)
        st.write(input_features)

    # for waveform in waveforms:
    #     st.audio(waveform, format="audio/wav")
    #     st.write("Sound uploaded")

model_all = load_subset_model(subset="all")
model_status = load_subset_model(subset="status")
model_appeal = load_subset_model(subset="appeal")
model_brand = load_subset_model(subset="brand_identity")
