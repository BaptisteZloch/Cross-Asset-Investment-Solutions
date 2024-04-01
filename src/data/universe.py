from functools import cached_property
from typing_extensions import Self
from typing import Callable, List, Optional
import pandas as pd
from collections.abc import Iterable

from utility.constants import TRADING_DAYS


class Universe:
    __universe: Optional[pd.DataFrame] = None
    _instance = None
    _PATH = "../data/data_cross_asset.xlsx"

    def __init__(
        self,
        market_column: str = "SPTR500N",
        keep_leveraged_etf: bool = False,
        keep_bonds_etf: bool = False,
        keep_only_benchmark_universe: bool = False,
    ) -> None:
        Universe.__load_universe()
        if keep_only_benchmark_universe is True and Universe.__universe is not None:
            Universe.__universe = Universe.__universe[["SX5T", "SPTR500N", "ESTR_ETF"]]
        else:
            col_to_remove = []
            if keep_bonds_etf is False:
                col_to_remove.extend(
                    [
                        "EURO_GOV_1-3Y",
                        "EURO_GOV_3-5Y",
                        "EURO_GOV_7-10Y",
                        "EURO_GOV_10-15Y",
                    ]
                )
            if keep_leveraged_etf is False:
                col_to_remove.extend(["NASDAQ-100_LEVIER_2", "SX5T_levier_2"])
            if len(col_to_remove) > 0 and Universe.__universe is not None:
                Universe.__universe = Universe.__universe.drop(columns=col_to_remove)
        assert (
            market_column in Universe.__universe.columns
        ), "Error, provide a value market column"
        self.__market_column = market_column

    @property
    def market_returns(self) -> pd.Series:
        """A pandas series representing the market returns, will be used to run the market regime algorithm on

        Returns:
            pd.Series: The Market returns.
        """
        return self.get_universe_returns()[self.__market_column]

    def get_market_returns(self) -> pd.Series:
        """A pandas series representing the market returns, will be used to run the market regime algorithm on

        Returns:
            pd.Series: The Market returns.
        """
        return self.market_returns

    @staticmethod
    def __load_universe():
        BASE_ETF = (
            pd.read_excel(
                Universe._PATH,
                sheet_name="ETF_bench",
                skiprows=[1],
                usecols=["Unnamed: 0", "SX5T", "SPTR500N", "ESTR_ETF"],
                index_col=0,
            )
            .dropna()
            .asfreq("B", method="ffill")
        )
        BASE_ETF["ESTR_ETF"] = (
            (BASE_ETF["ESTR_ETF"] / TRADING_DAYS / 100) + 1
        ).cumprod()
        ETF_THEMATICS = (
            pd.read_excel(
                Universe._PATH,
                sheet_name="Thematiques_others",
                usecols=[
                    "Unnamed: 0",
                    "EUROPE _VALUE_FACTOR",
                    # "EUROPE _MOMENTUM_FACTOR",
                    "WATER_ESG",
                    "STOXX_EUROPE 600_TECHNOLOGY",
                    "STOXX_EUROPE 600_HEALTHCARE",
                    "EURO_GOV_1-3Y",
                    "EURO_GOV_3-5Y",
                    "EURO_GOV_7-10Y",
                    "EURO_GOV_10-15Y",
                    "EPSILON_TREND",
                ],
                index_col=0,
                skiprows=[1],
            )
            .dropna()
            .asfreq("B", method="ffill")
        )
        LEVERAGED_ETF = (
            pd.read_excel(
                Universe._PATH,
                sheet_name="ETF_levier",
                skiprows=[1],
                usecols=["Unnamed: 0", "SX5T_levier_2", "NASDAQ-100_LEVIER_2"],
                index_col=0,
            )
            .dropna()
            .asfreq("B", method="ffill")
        )
        FUTURES = (
            pd.read_excel(
                "../data/data_cross_asset.xlsx",
                sheet_name="Futurs",
                usecols=["Date", "Px fut SX5E", "Px fut sp500", "Px fut nasdaq"],
                index_col=0,
            )
            .dropna()
            .asfreq("B", method="ffill")
        )
        Universe.__universe = pd.merge(
            ETF_THEMATICS,
            pd.merge(
                LEVERAGED_ETF,
                pd.merge(
                    FUTURES, BASE_ETF, left_index=True, right_index=True, how="inner"
                ),
                left_index=True,
                right_index=True,
                how="inner",
            ),
            left_index=True,
            right_index=True,
            how="inner",
        )

    @staticmethod
    def __check_loaded(func: Callable):
        def wrapper(*args, **kwargs):
            if Universe.__universe is None:
                Universe.__load_universe()
            return func(*args, **kwargs)

        return wrapper

    @property
    def defensive_securities(self) -> List[str]:
        return list(
            set(Universe.__universe.columns.to_list()).intersection(
                [
                    "EUROPE _VALUE_FACTOR",
                    "STOXX_EUROPE 600_HEALTHCARE",
                    "EPSILON_TREND",
                    "Px fut SX5E",
                    "Px fut sp500",
                    "SX5T",
                    "SPTR500N",
                    "ESTR_ETF",
                ]
            )
        )

    @property
    def offensive_securities(self) -> List[str]:
        return list(
            set(Universe.__universe.columns.to_list()).intersection(
                [
                    "WATER_ESG",
                    "STOXX_EUROPE 600_TECHNOLOGY",
                    "EPSILON_TREND",
                    "Px fut SX5E",
                    "Px fut sp500",
                    "Px fut nasdaq",
                    "SX5T",
                    "SPTR500N",
                    "ESTR_ETF",
                ]
            )
        )

    @staticmethod
    # @__check_loaded
    def get_universe_price():
        return Universe.__universe

    @staticmethod
    # @__check_loaded
    def get_universe_returns():
        return Universe.__universe.pct_change().fillna(0)

    @staticmethod
    # @__check_loaded
    def get_universe_perfs():
        return (Universe.__universe.pct_change().fillna(0) + 1).cumprod()

    @staticmethod
    # @__check_loaded
    def get_universe_securities() -> List[str]:
        return Universe.__universe.columns.to_list()

    def __new__(cls, *args, **kwargs) -> Self:
        """Singleton pattern implementation.

        Returns:
            Self: The unique instance of the class.
        """
        if cls._instance is None:
            cls._instance = super(Universe, cls).__new__(cls)
        return cls._instance
