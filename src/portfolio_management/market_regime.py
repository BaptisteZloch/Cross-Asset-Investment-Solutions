from typing import Dict, Literal
import numpy as np
import numpy.typing as npt

from hmmlearn.hmm import GaussianHMM
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def normalize_regime(signal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    proportion = np.mean(signal)
    if proportion >= 0.5:
        return -1 * signal + 1
    return signal


def detect_market_regime(
    market_data: npt.NDArray[np.float32],
    scale_data: bool = True,
    scaler: Literal["robust", "standard", "minmax"] = "robust",
    *args,
    **kwargs
) -> npt.NDArray[np.float32]:
    if len(market_data.shape) == 1:
        market_data = market_data.reshape(-1, 1)
    if scale_data is True:
        SCALER_STRING_TO_SCALER: Dict[str, TransformerMixin] = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(feature_range=(-1, 1)),
        }
        X = SCALER_STRING_TO_SCALER[scaler].fit_transform(market_data)
    else:
        X = market_data
    HMM_MODEL = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        algorithm=kwargs.get("algorithm", "viterbi"),
        random_state=42,
        n_iter=100,
        tol=0.1,
        verbose=False,
        implementation="log",
    )

    HMM_MODEL.fit(X)
    return normalize_regime(HMM_MODEL.predict(X))
