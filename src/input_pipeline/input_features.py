import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.preprocessing import MinMaxScaler


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
    timbre_pca = joblib.load("data/models/input_feature_models/timbre_pca.pkl")
    timbre_scaler = joblib.load("data/models/input_feature_models/timbre_scaler.pkl")
    timbre_gmm = joblib.load(
        "data/models/input_feature_models/timbre_gmm.pkl"
    ).best_estimator_
    timbre_lda = joblib.load(
        "data/models/input_feature_models/timbre_coherence_lda.pkl"
    ).best_estimator_
    return timbre_pca, timbre_scaler, timbre_gmm, timbre_lda


def get_model_input_timbre(waveform, global_variables) -> np.array:
    """
    Extract timbre features from an audio waveform and return
    topic distribution for 16 timbre topics.
    1) Extract+combine MFCC and zero-crossing rate features (per frame)
    2) Find timbre clusters using GMM (per frame) -> get cluster count (per sound)
    3) Transform cluster count to timbre topics using LDA
    """
    timbre_pca, timbre_scaler, timbre_gmm, timbre_lda = load_timbre_models()
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

    # cluster count per sound is used as input for LDA
    timbre_clusters = get_cluster_count(
        timbre_feature, timbre_gmm, pca=timbre_pca, scaler=timbre_scaler
    )
    timbre_topics = timbre_lda.transform(timbre_clusters.reshape(1, -1))[0]
    return timbre_topics
