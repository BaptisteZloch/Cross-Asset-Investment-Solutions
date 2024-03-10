from typing import Callable, List, Optional, Self
import pandas as pd
from collections.abc import Iterable


class Universe:
    __universe: Optional[pd.DataFrame] = None
    _instance = None

    @staticmethod
    def __load_universe():
        Universe.__universe = pd.read_csv(
            "../data/universe.csv", parse_dates=True, index_col="date"
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
