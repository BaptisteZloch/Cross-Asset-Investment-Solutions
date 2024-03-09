from typing import Dict, Literal
import numpy as np
import numpy.typing as npt

from hmmlearn.hmm import GaussianHMM
from sklearn.base import TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_regime(signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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
    """_summary_

    Args:
        market_data (npt.NDArray[np.float32]): _description_
        scale_data (bool, optional): _description_. Defaults to True.
        scaler_type (Literal[&quot;robust&quot;, &quot;standard&quot;, &quot;minmax&quot;], optional): _description_. Defaults to "standard".

    Returns:
        npt.NDArray[np.float32]: _description_
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
