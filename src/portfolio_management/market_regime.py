from typing import Dict, Literal
import numpy as np
import numpy.typing as npt

from hmmlearn.hmm import GaussianHMM
from sklearn.base import TransformerMixin
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from utility.types import RegimeDetectionModels
from utility.utils import normalize_regime


def detect_market_regime(
    market_data: npt.NDArray[np.float32],
    market_regime_detection_algorithm: RegimeDetectionModels = RegimeDetectionModels.HIDDEN_MARKOV_MODEL,
    scale_data: bool = True,
    scaler_type: Literal["robust", "standard", "minmax"] = "standard",
    *args,
    **kwargs
) -> npt.NDArray[np.float32]:
    """Detect market regime given a dataset and an algorithm. This process in offline which means it uses the whole dataset for prediction and training.

    Args:
        market_data (npt.NDArray[np.float32]): The vector corresponding to the market on which we want to cluster the regimes.
        market_regime_detection_algorithm (RegimeDetectionModels): The desired model to use for market regime detection.
        scale_data (bool, optional): Whether to scale the data or not using the scaler_type. Defaults to True.
        scaler_type (Literal[&quot;robust&quot;, &quot;standard&quot;, &quot;minmax&quot;], optional): The scaler type used to scale the data. If scale_data is False this argument is ignored. Defaults to "standard".
        **kwargs: "algorithm" in {"viterbi", "map"}
    Returns:
        npt.NDArray[np.float32]: The regime detected on the market data.
    """
    if len(market_data.shape) == 1:
        market_data = market_data.reshape(-1, 1)
    if scale_data is True:
        SCALER_STRING_TO_SCALER: Dict[str, TransformerMixin] = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(feature_range=(-1, 1)),
        }
        X = SCALER_STRING_TO_SCALER[scaler_type].fit_transform(market_data)
    else:
        X = market_data
    if market_regime_detection_algorithm == RegimeDetectionModels.HIDDEN_MARKOV_MODEL:
        MODEL = GaussianHMM(
            n_components=2,
            covariance_type="full",
            algorithm=kwargs.get("algorithm", "viterbi"),
            random_state=42,
            n_iter=100,
            tol=0.001,
            verbose=False,
            implementation="log",
        )
    elif market_regime_detection_algorithm == RegimeDetectionModels.GAUSSIAN_MIXTURE:
        MODEL = GaussianMixture(
            n_components=2,
            covariance_type="full",
            init_params="k-means++",
            random_state=42,
            verbose=0,
            max_iter=100,
        )
    elif market_regime_detection_algorithm == RegimeDetectionModels.KMEANS:
        MODEL = KMeans(
            n_clusters=2, init="k-means++", random_state=42, verbose=0, max_iter=100
        )
    elif market_regime_detection_algorithm == RegimeDetectionModels.BISECTING_KMEANS:
        MODEL = BisectingKMeans(
            n_clusters=2,
            init="k-means++",
            random_state=42,
            verbose=0,
            max_iter=100,
            bisecting_strategy="largest_cluster",
        )
    else:
        raise ValueError(
            "Provide a valid market_regime_detection_algorithm among : hmm, gaussian_mixture"
        )
    MODEL.fit(X)
    return normalize_regime(MODEL.predict(X))
