from typing import Callable, List, Optional, Self
import pandas as pd
from collections.abc import Iterable


class Universe:
    __universe: Optional[pd.DataFrame] = None
    _instance = None
    _PATH = "../data/data_cross_asset.xlsx"

    def __init__(
        self,
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
        ETF_THEMATICS = (
            pd.read_excel(
                Universe._PATH,
                sheet_name="Thematiques_others",
                usecols=[
                    "Unnamed: 0",
                    "EUROPE _VALUE_FACTOR",
                    "EUROPE _MOMENTUM_FACTOR",
                    "WATER_ESG",
                    "STOXX_EUROPE 600_TECHNOLOGY",
                    "STOXX_EUROPE 600_HEALTHCARE",
                    "EURO_GOV_1-3Y",
                    "EURO_GOV_3-5Y",
                    "EURO_GOV_7-10Y",
                    "EURO_GOV_10-15Y",
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

    @staticmethod
    @__check_loaded
    def get_universe_price():
        return Universe.__universe

    @staticmethod
    @__check_loaded
    def get_universe_returns():
        return Universe.__universe.pct_change().fillna(0)

    @staticmethod
    @__check_loaded
    def get_universe_perfs():
        return (Universe.__universe.pct_change().fillna(0) + 1).cumprod()

    @staticmethod
    @__check_loaded
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
