from typing_extensions import Self
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from utility.constants import TRADING_DAYS
from utility.types import RebalanceFrequencyEnum
from utility.utils import compute_weights_drift, get_rebalance_dates


class Benchmark:
    _PATH = "../data/data_cross_asset.xlsx"
    __BASE_WEIGHTS = {"OISESTR": 0.3, "SPX": 0.2, "SX5T": 0.5}
    __benchmark_returns: Optional[pd.Series] = None
    __weights_df: Optional[pd.DataFrame] = None
    __benchmark_perf: Optional[pd.Series] = None
    _instance = None

    def __init__(
        self,
        rebalance_frequency: RebalanceFrequencyEnum = RebalanceFrequencyEnum.MONTH_END,
    ) -> None:
        self.__benchmark_components_returns = self.get_benchmark_returns_data()
        self.__rebalance_frequency = rebalance_frequency

    def __construct_benchmark_history(self):
        returns_histo, weights_histo = [], []
        REBALANCE_DATES = get_rebalance_dates(
            start_date=self.__benchmark_components_returns.index[0],
            end_date=self.__benchmark_components_returns.index[-1],
            frequency=self.__rebalance_frequency,
        )
        VERBOSE = False
        for index, row in tqdm(
            self.__benchmark_components_returns.iterrows(),
            desc="Creating benchmark...",
            total=self.__benchmark_components_returns.shape[0],
            leave=False,
        ):
            if index in REBALANCE_DATES:
                if VERBOSE:
                    print(f"Rebalancing the portfolio on {index}...")
                weights = self.__BASE_WEIGHTS
            weights_histo.append(weights)

            returns_histo.append(
                (
                    self.__benchmark_components_returns[list(weights.keys())]
                    .loc[index]  # type: ignore
                    .to_numpy()
                    @ np.array(list(weights.values()))
                )
            )
            weights = compute_weights_drift(
                list(weights.keys()),
                np.array(list(weights.values())),
                self.__benchmark_components_returns[list(weights.keys())]
                .loc[index]  # type: ignore
                .to_numpy(),
            )
        # The returns of the benchmark
        self.__benchmark_returns = pd.Series(
            returns_histo, index=self.__benchmark_components_returns.index, dtype=float
        )
        # The cumulative returns of the benchmark
        self.__benchmark_perf = (self.__benchmark_returns.copy() + 1).cumprod()
        # The weights of the benchmark
        self.__weights_df = pd.DataFrame(
            weights_histo, index=self.__benchmark_components_returns.index, dtype=float
        ).fillna(0)

    @property
    def benchmark_returns(self) -> pd.Series:
        if self.__benchmark_returns is None:
            self.__construct_benchmark_history()
        self.__benchmark_returns.name = "benchmark_returns"
        return self.__benchmark_returns

    @property
    def benchmark_weights(self) -> pd.DataFrame:
        if self.__weights_df is None:
            self.__construct_benchmark_history()
        return self.__weights_df

    @property
    def benchmark_perf(self) -> pd.Series:
        if self.__benchmark_perf is None:
            self.__construct_benchmark_history()
        self.__benchmark_perf.name = "benchmark_perf"
        return self.__benchmark_perf

    def get_benchmark_price_data(self) -> pd.DataFrame:
        benchmark = (
            pd.read_excel(
                self._PATH,
                sheet_name="ETF_bench",
                skiprows=[1],
                usecols=["Unnamed: 0", "SX5T", "SPTR500N", "ESTR_ETF"],
                index_col=0,
            )
            .rename(
                columns={
                    "ESTR_ETF": "OISESTR",
                    "SPTR500N": "SPX",
                }
            )
            .dropna()
            .asfreq("B", method="ffill")
        )
        benchmark["OISESTR"] = (
            (benchmark["OISESTR"] / TRADING_DAYS / 100) + 1
        ).cumprod()
        return benchmark

    def get_benchmark_returns_data(self) -> pd.DataFrame:
        benchmark = self.get_benchmark_price_data().pct_change().fillna(0)
        return benchmark[list(self.__BASE_WEIGHTS.keys())]

    def __new__(cls, *args, **kwargs) -> Self:
        """Singleton pattern implementation.

        Returns:
            Self: The unique instance of the class.
        """
        if cls._instance is None:
            cls._instance = super(Benchmark, cls).__new__(cls)
        return cls._instance
