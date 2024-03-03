from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from portfolio_management.allocation import ALLOCATION_DICT
from utility.types import AllocationMethodsEnum, RebalanceFrequencyEnum

from utility.utils import compute_weights_drift, get_rebalance_dates


class Backtester:
    def __init__(self, universe_returns: pd.DataFrame) -> None:
        self.__universe_returns = universe_returns

    def run_backtest(
        self,
        allocation_type: AllocationMethodsEnum,
        rebalance_frequency: RebalanceFrequencyEnum,
        transaction_cost_by_securities: Dict[str, float],
        bullish_leverage_by_securities: Optional[Dict[str, float]] = None,
        bearish_leverage_by_securities: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        starting_offset: int = 20,
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        assert starting_offset >= 0, "Error, provide a positive starting offset."
        assert set(transaction_cost_by_securities.keys()) == set(
            self.__universe_returns.columns
        ), "Error, you need to provide transaction cost for every security in the universe"

        # List to store the returns and weights at each iteration (i.e. Days)
        returns_histo, weights_histo = [], []

        # Get all rebalance dates during the backtest
        REBALANCE_DATES = get_rebalance_dates(
            start_date=self.__universe_returns.index[0],
            end_date=self.__universe_returns.index[-1],
            frequency=rebalance_frequency,
        )

        first_rebalance = False  # Create a portfolio at the first date

        for index, row in tqdm(
            self.__universe_returns.iloc[starting_offset:].iterrows(),
            desc="Backtesting portfolio...",
            total=self.__universe_returns.shape[0],
            leave=False,
        ):
            if index in REBALANCE_DATES or first_rebalance is False:
                first_rebalance = True
                if verbose:
                    print(f"Rebalancing the portfolio on {index}...")
                weights = ALLOCATION_DICT[allocation_type](
                    self.__universe_returns.columns, self.__universe_returns.loc[:index]
                )
                # Row returns with applied fees
                returns_with_applied_fees = []
                for ind, value in row.loc[list(weights.keys())].items():
                    returns_with_applied_fees.append(
                        value - transaction_cost_by_securities.get(str(ind))
                    )
                returns = np.array(returns_with_applied_fees)
            else:
                # Row returns
                returns = row.loc[list(weights.keys())].to_numpy()
            # Append the current weights to the list
            weights_histo.append(weights)
            # Create numpy weights for matrix operations
            weights_np = np.array(list(weights.values()))
            # Append the current return to the list
            returns_histo.append((returns @ weights_np))

            # Compute the weight drift due to assets price fluctuations
            weights = compute_weights_drift(
                row.loc[list(weights.keys())].index.to_list(),
                weights_np,
                returns,
            )

        # The returns of the ptf
        ptf_returns = pd.Series(
            returns_histo,
            index=self.__universe_returns.iloc[starting_offset:].index,
            dtype=float,
            name="ptf_returns",
        )

        # The weights of the ptf
        ptf_weights_df = pd.DataFrame(
            weights_histo,
            index=self.__universe_returns.iloc[starting_offset:].index,
            dtype=float,
        ).fillna(0)
        return ptf_returns, ptf_weights_df
