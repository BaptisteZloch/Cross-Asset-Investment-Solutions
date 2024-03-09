from typing import Dict, Literal
import numpy as np
import numpy.typing as npt

from hmmlearn.hmm import GaussianHMM
from sklearn.base import TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_regime(signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Function that normalize the regime with 1 = Bearish regime and 0 bullish regime. historically the proportion of bullish regime is higher. We applied this rule here.

    Args:
        signal (npt.NDArray[np.float32]): The 1 and 0 signal detected with the detection algorithm.

    Returns:
        npt.NDArray[np.float32]: The normalized regime
    """
    proportion = np.mean(signal)
    if proportion >= 0.5:
        return -1 * signal + 1
    return signal


def detect_market_regime(
    market_data: npt.NDArray[np.float32],
    market_regime_detection_algorithm: Literal["hmm", "gaussian_mixture", "jump_model"],
    scale_data: bool = True,
    scaler_type: Literal["robust", "standard", "minmax"] = "standard",
    *args,
    **kwargs
) -> npt.NDArray[np.float32]:
    """Detect market regime given a dataset and an algorithm. This process in offline which means it uses the whole dataset for prediction and training.

    Args:
        market_data (npt.NDArray[np.float32]): The vector corresponding to the market on which we want to cluster the regimes.
        scale_data (bool, optional): Whether to scale the data or not using the scaler_type. Defaults to True.
        scaler_type (Literal[&quot;robust&quot;, &quot;standard&quot;, &quot;minmax&quot;], optional): The scaler type used to scale the data. If scale_data is False this argument is ignored. Defaults to "standard".

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
    if market_regime_detection_algorithm == "hmm":
        HMM_MODEL = GaussianHMM(
            n_components=2,
            covariance_type="full",
            algorithm=kwargs.get("algorithm", "viterbi"),
            random_state=42,
            n_iter=100,
            tol=0.001,
            verbose=False,
            implementation="log",
        )

        HMM_MODEL.fit(X)
        return normalize_regime(HMM_MODEL.predict(X))
    elif market_regime_detection_algorithm == "gaussian_mixture":
        GM_MODEL = GaussianMixture(
            n_components=2,
            covariance_type="full",
            init_params="k-means++",
            random_state=True,
            verbose=0,
            max_iter=100,
        )

        GM_MODEL.fit(X)
        return normalize_regime(GM_MODEL.predict(X))
    elif market_regime_detection_algorithm == "jump_model":
        raise NotImplementedError()
    else:
        raise ValueError(
            "Provide a valid market_regime_detection_algorithm among : hmm, gaussian_mixture, jump_model"
        )
