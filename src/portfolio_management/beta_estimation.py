from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from filterpy.kalman import KalmanFilter
from tqdm import tqdm


def beta_convergence(
    current_value: Union[float, npt.NDArray[np.float32]],
    initial_value: float,
    long_term_value: float,
    smoothing_lambda: float,
) -> Union[float, npt.NDArray[np.float32]]:
    """_summary_

    Args:
        current_value ( Union[float,npt.NDArray[np.float32]]): The current beta exposure calculated
        initial_value (float): The initial value (starting point)
        long_term_value (float): The long target to converge to.
        smoothing_lambda (float): The smoothing coefficient, the higher the more slowly the value will converge.

    Returns:
        Union[float, npt.NDArray[np.float32]]: The new beta calculated or a vector of each value of the series.
    """
    assert (
        0.1 <= long_term_value <= 3 and 0.1 <= initial_value <= 3
    ), "long_term_value must be between 0.1 and 3"
    return long_term_value - (
        (long_term_value - initial_value)
        * (1 - np.exp(-smoothing_lambda / current_value))
    )


def estimate_dynamic_beta_and_alpha(
    market_returns: npt.NDArray[np.float32], asset_returns: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], ...]:
    """Using a Linear Kalman filter this function estimates beta and alpha dynamically over time.

    Args:
        market_returns (npt.NDArray[np.float32]): The market returns used in the CAPM.
        asset_returns (npt.NDArray[np.float32]): The asset returns to calculate the beta with the benchmark.

    Returns:
        Tuple[npt.NDArray[np.float32], ...]: 2 numpy arrays : one with the alpha and one with the beta
    """
    assert (
        market_returns.shape[0] == asset_returns.shape[0]
    ), "Error provide market and asset return with the same shape."
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.01, 1])  # Initial state (initial guess for alpha and beta)
    kf.P = np.array([[1, 0], [0, 1]])  # Initial state covariance matrix

    kf.F = np.array([[1, 0], [0, 1]])
    kf.Q = np.array([[0.1, 0], [0, 1]])  # Covariance process
    # kf.H = np.array([[1, 0]])
    kf.R = np.array([[0.1, 0], [0, 0.001]])  # Covariance measure

    estimated_beta, estimated_alpha = [], []
    for asset_return, market_return in tqdm(
        zip(asset_returns, market_returns),
        total=asset_returns.shape[0],
        leave=False,
        desc="Computing estimates",
    ):
        kf.predict()
        kf.update(z=asset_return, H=np.array([[1, market_return], [0, 0]]))
        estimated_beta.append(kf.x[-1])
        estimated_alpha.append(kf.x[0])
    return np.array(estimated_alpha), np.array(estimated_beta)


def predict_next_beta_and_alpha(
    market_returns: npt.NDArray[np.float32], asset_returns: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Using a Linear Kalman filter this function estimates beta and alpha dynamically over time and predict the next state (without correction).

    Args:
        market_returns (npt.NDArray[np.float32]): The market returns used in the CAPM.
        asset_returns (npt.NDArray[np.float32]): The asset returns to calculate the beta with the benchmark.

    Returns:
        npt.NDArray[np.float32]: A vector with estimates for the next alpha first and beta.
    """
    assert (
        market_returns.shape[0] == asset_returns.shape[0]
    ), "Error provide market and asset return with the same shape."
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0.01, 1])  # Initial state (initial guess for alpha and beta)
    kf.P = np.array([[1, 0], [0, 1]])  # Initial state covariance matrix

    kf.F = np.array([[1, 0], [0, 1]])
    kf.Q = np.array([[0.1, 0], [0, 1]])  # Covariance process
    # kf.H = np.array([[1, 0]])
    kf.R = np.array([[0.1, 0], [0, 0.001]])  # Covariance measure

    estimated_beta, estimated_alpha = [], []
    for asset_return, market_return in tqdm(
        zip(asset_returns, market_returns),
        total=asset_returns.shape[0],
        leave=False,
        desc="Computing estimates",
    ):
        kf.predict()
        kf.update(z=asset_return, H=np.array([[1, market_return], [0, 0]]))
        estimated_beta.append(kf.x[-1])
        estimated_alpha.append(kf.x[0])
    # kf.predict()
    return kf.x


def historical_rolling_beta(
    market_returns: npt.NDArray[np.float32],
    asset_returns: npt.NDArray[np.float32],
    window_size: int = 252,
) -> npt.NDArray[np.float32]:
    """Estimates dynamic beta using rolling window OLS regression.

    Args:
        market_returns (npt.NDArray[np.float32]): The market returns used in the CAPM.
        asset_returns (npt.NDArray[np.float32]): The asset returns to calculate the beta with the benchmark.
        window_size (int, optional): The past window length used to calculate the beta. Defaults to 252.

    Returns:
        npt.NDArray[np.float32]: The historical estimated beta
    """
    beta_estimates = []
    for i in range(window_size, len(market_returns)):
        window_market_return = market_returns[i - window_size : i]
        window_asset_return = asset_returns[i - window_size : i]
        # OLS regression to estimate beta
        A = np.vstack([window_market_return, np.ones(len(window_market_return))]).T
        beta, _ = np.linalg.lstsq(A, window_asset_return, rcond=None)[0]
        beta_estimates.append(beta)
    return np.hstack(
        (
            np.array([beta_estimates[0] for _ in range(window_size)]),
            np.array(beta_estimates),
        )
    )
