import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.preprocessing import MinMaxScaler
from mosqito.sq_metrics import loudness_zwtv

# disable user warnings for sklearn
import warnings

warnings.filterwarnings("ignore")


def normalize_feature(
    feature: np.array, min_value: float, max_value: float
) -> np.array:
    """
    Normalize a feature array to the range [0, 1]
    """
    feature = (feature - min_value) / (max_value - min_value)
    return feature


def get_normalized_pca_feature(feature_array: np.array, pca_model) -> np.array:
    """
    Normalize and reduce the dimensionality of a feature array using PCA.
    The Kaiser criterion is applied to select the number of principal components.
    Only dimensions with eigenvalues greater than 1 are kept.
    """
    feature_array_pca = pca_model.transform(feature_array)
    feature_array_pca = feature_array_pca[:, :8]
    scaler = MinMaxScaler()
    normalized_pca_features = scaler.fit_transform(feature_array_pca)
    return normalized_pca_features


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    # as GridSearchCV is designed to maximize a score
    # (maximizing the negative BIC is equivalent to minimizing the BIC).
    return -estimator.bic(X)


def get_cluster_count(feature_array, gmm, pca=None, scaler=None) -> np.array:
    """
    Get the cluster count of a feature array using a GMM model.
    If PCA and scaler are provided, the feature array is first transformed and scaled.
    """

    if pca is not None and scaler is not None:
        feature_array_pca = pca.transform(feature_array)[:, :8]
        feature_array_pca = scaler.transform(feature_array_pca)
    else:
        feature_array_pca = feature_array
    clusters = gmm.predict(feature_array_pca)
    cluster_count = np.bincount(clusters)
    # make sure cluster_count has the same length as the number of clusters -> zero fill
    if len(cluster_count) < gmm.n_components:
        cluster_count = np.concatenate(
            (cluster_count, np.zeros(gmm.n_components - len(cluster_count)))
        )
    return cluster_count.astype(int)


def extract_mfcc_features(
    waveform, n_mfccs, sample_rate, frame_size, hop_size
) -> np.array:
    """
    Wrapper for the librosa mfcc function.
    """
    return librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfccs,
        n_fft=frame_size,
        hop_length=hop_size,
    )


def extract_zero_crossing_rate(waveform, frame_length, hop_length) -> np.array:
    """
    Wrapper for the librosa zero_crossing_rate function.
    """
    return librosa.feature.zero_crossing_rate(
        waveform, frame_length=frame_length, hop_length=hop_length
    )


def create_timbre_feature(mfcc: np.array, zcr: np.array) -> np.array:
    """
    Timbre feature: per frame the zero-crossing rate and 24 MFCC features
    are concatenated.
    """
    assert mfcc.shape[1] == zcr.shape[1]
    timbre_feature = np.concatenate((zcr, mfcc), axis=0)
    return timbre_feature


def load_timbre_models():
    """
    Load the timbre feature models.
    PCA, Scaler and GMM for clusters and LDA for topics.
    """
    timbre_pca = joblib.load("./models/input_feature_models/timbre_pca.pkl")
    timbre_scaler = joblib.load("./models/input_feature_models/timbre_scaler.pkl")
    timbre_gmm = joblib.load("./models/input_feature_models/timbre_gmm_best_est.pkl")
    timbre_lda = joblib.load("./models/input_feature_models/timbre_lda_best_est.pkl")
    return timbre_pca, timbre_scaler, timbre_gmm, timbre_lda


def get_model_input_timbre(waveform, global_variables) -> np.array:
    """
    Extract timbre features from an audio waveform and return
    topic distribution for 16 timbre topics.
    1) Extract+combine MFCC and zero-crossing rate features (per frame)
    2) Find timbre clusters using GMM (per frame) -> get cluster count (per sound)
    3) Transform cluster count to timbre topics using LDA
    """
    # extract audio features
    mfcc = extract_mfcc_features(
        waveform,
        global_variables["n_mfcc"],
        global_variables["sample_rate"],
        global_variables["frame_size"],
        global_variables["hop_size"],
    )
    mfcc = normalize_feature(
        mfcc, global_variables["mfcc_min"], global_variables["mfcc_max"]
    )
    zcr = extract_zero_crossing_rate(
        waveform, global_variables["frame_size"], global_variables["hop_size"]
    )
    zcr = normalize_feature(
        zcr, global_variables["zcr_min"], global_variables["zcr_max"]
    )
    timbre_feature = create_timbre_feature(mfcc, zcr).T

    timbre_pca, timbre_scaler, timbre_gmm, timbre_lda = load_timbre_models()
    # cluster count per sound is used as input for LDA
    timbre_clusters = get_cluster_count(
        timbre_feature, timbre_gmm, pca=timbre_pca, scaler=timbre_scaler
    )
    timbre_topics = timbre_lda.transform(timbre_clusters.reshape(1, -1))[0]
    return timbre_topics


def extract_chroma_features(waveform, sample_rate, frame_size, hop_size) -> np.array:
    """
    Wrapper for the librosa chroma_stft function.
    """
    return librosa.feature.chroma_stft(
        y=waveform, sr=sample_rate, n_fft=frame_size, hop_length=hop_size
    )


def get_clean_chroma(chroma, waveform, hop_size) -> np.array:
    """
    1) mask where waveform < 40dB
    2) smoothe chroma with median filter (3x3)
    3) apply thresholding to remove noise (0.8)
    """

    non_zero_intervals = librosa.effects.split(y=waveform, top_db=40) // hop_size
    mask = np.zeros(chroma.shape)
    for interval in non_zero_intervals:
        mask[:, interval[0] : interval[1]] = 1
    non_zero_chroma = chroma * mask

    median_chroma = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(3) / 3, mode="same"),
        axis=1,
        arr=non_zero_chroma,
    )

    threshold_chroma = median_chroma.copy()
    threshold_chroma[median_chroma < 0.8] = 0
    return threshold_chroma


def custom_chroma_argmax(x) -> np.array:
    """
    Works like np.argmax but returns -1 if
    the array is all zeros in the frame
    """
    argmax = []
    for idx in range(x.shape[1]):
        x_slice = x[:, idx]
        if not x_slice.any() == 0:
            argmax.append(np.argmax(x_slice))
        else:
            argmax.append(-1)
    return np.array(argmax)


def get_relative_chroma(chroma: np.array) -> np.array:
    """
    1) determine dominant chroma
    2) shift each chroma frame so that dominant chroma = 0
    """
    chroma_argmax = custom_chroma_argmax(chroma)
    # only count frames where chroma is present (-1 means frame is too quiet)
    dominant_chroma = np.argmax(np.bincount(chroma_argmax[chroma_argmax != -1]))
    chroma_argmax[chroma_argmax != -1] = (
        chroma_argmax[chroma_argmax != -1] - dominant_chroma
    ) % 12
    relative_chroma = np.array(chroma_argmax)
    return relative_chroma


def load_chroma_model():
    """
    Load the chroma feature models.
    LDA for topics.
    """
    chroma_lda = joblib.load("./models/input_feature_models/chroma_lda_best_est.pkl")
    return chroma_lda


def get_chroma_count(relative_chroma) -> np.array:
    """
    Count the number of frames for each chroma value.
    Used as the input for the chroma LDA model.
    """
    # 12 semitones + 1 for no chroma (-1)
    chroma_count = np.zeros(13)
    for val in relative_chroma:
        idx = val + 1  # Shift values to align with index
        chroma_count[idx] += 1

    return chroma_count.astype(int)


def get_model_input_chroma(waveform, global_variables) -> np.array:
    """
    Extract chroma features from an audio waveform and return
    topic distribution for 12 chroma topics.
    1) Extract chroma features
    2) Clean chroma and shift so that dominant chroma = 0 (relative chroma)
    3) Transform chroma count to chroma topics using LDA
    """

    chroma = extract_chroma_features(
        waveform,
        global_variables["sample_rate"],
        global_variables["frame_size"],
        global_variables["hop_size"],
    )
    # no normalization needed for chroma features (between 0 and 1 per default)
    clean_chroma = get_clean_chroma(chroma, waveform, global_variables["hop_size"])
    relative_chroma = get_relative_chroma(clean_chroma)

    chroma_count = get_chroma_count(relative_chroma)
    chroma_lda = load_chroma_model()
    chroma_topics = chroma_lda.transform(chroma_count.reshape(1, -1))[0]
    return chroma_topics


def extract_loudness_feature(waveform, sample_rate):
    """
    Wrapper for the loudness_zwtv function from the mosqito library.
    This is a subjective loudness measuer suitable for short
    non-stationary signals.
    """
    loudness, _, _, _ = loudness_zwtv(waveform, sample_rate)
    return loudness[::4]


def load_loudnes_models():
    """
    Load the loudness feature models.
    GMM for clusters and LDA for topics.
    """
    loudness_gmm = joblib.load(
        "./models/input_feature_models/loudness_gmm_best_est.pkl"
    )
    loudness_lda = joblib.load(
        "./models/input_feature_models/loudness_lda_best_est.pkl"
    )
    return loudness_gmm, loudness_lda


def get_model_input_loudness(waveform, global_variables) -> np.array:
    """
    Extract loudness features from an audio waveform and return
    topic distribution for 8 loudness topics.
    1) Extract Zwickel loudness features
    2) Find loudness clusters using GMM -> get cluster count
    3) Transform cluster count to loudness topics using LDA
    """
    loudness_gmm, loudness_lda = load_loudnes_models()
    loudness = extract_loudness_feature(waveform, global_variables["sample_rate"])

    normalized_loudness = normalize_feature(
        loudness, global_variables["loudness_min"], global_variables["loudness_max"]
    ).reshape(-1, 1)
    cluster_count = get_cluster_count(normalized_loudness, loudness_gmm)
    loudness_topics = loudness_lda.transform(cluster_count.reshape(1, -1))[0]
    return loudness_topics


def oh_enc_industry(industry: str, industry_list: list) -> np.array:
    """
    One-Hot-Encode the industry feature
    ["Apps", "Consumer", "Future", "Health", "Home", "Mobility", "Os"]
    """
    assert industry in industry_list, f"{industry} not in {industry_list}"
    industry_idx = industry_list.index(industry)
    industry_oh = np.zeros(len(industry_list))
    industry_oh[industry_idx] = 1
    return industry_oh


def get_model_input(waveform: np.array, global_variables) -> np.array:
    """
    Extract all input features for the models.
    16 timbre topics, 12 chroma topics, 8 loudness topics
    """

    # all models use musical features as input
    timbre_topics = get_model_input_timbre(waveform, global_variables)
    chroma_topics = get_model_input_chroma(waveform, global_variables)
    loudness_topics = get_model_input_loudness(waveform, global_variables)
    return np.concatenate((timbre_topics, chroma_topics, loudness_topics))


def append_industry_to_model_input(
    model_input: np.array, industry: str, global_variables: dict
) -> np.array:
    """
    Append the industry feature to the model input. If users decide to include
    the industry feature this is done after calculating the input features.
    Otherwise feature extraction had to be redone and this takes time.
    """
    assert (
        industry in global_variables["industry_list"]
    ), f"Industry '{industry}' not in {global_variables['industry_list']}"
    industry_oh = oh_enc_industry(industry, global_variables["industry_list"])
    return np.concatenate((model_input, industry_oh))
