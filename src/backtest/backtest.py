from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from backtest.metrics import drawdown
from backtest.reports import plot_from_trade_df, print_portfolio_strategy_report
from data.universe import Universe
from portfolio_management.allocation import ALLOCATION_DICT, Allocation
from portfolio_management.beta_estimation import predict_next_beta_and_alpha
from portfolio_management.market_regime import detect_market_regime
from utility.types import (
    AllocationMethodsEnum,
    RebalanceFrequencyEnum,
    RegimeDetectionModels,
)

from utility.utils import (
    compute_weights_drift,
    get_rebalance_dates,
    get_regime_detection_dates,
)


class Backtester:
    def __init__(
        self,
        universe_returns: pd.DataFrame,
        market_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> None:
        """Constructor for the backtester class

        Args:
        ----
            universe_returns (pd.DataFrame): The returns of the universe each columns is an asset, each row represent a date and returns for the assets. The DataFrame must have a `DatetimeIndex` with freq=B.
            market_returns (pd.Series): The returns of the market on which the regime will be detected. The Series must have a `DatetimeIndex` with freq=B.
            benchmark_returns (pd.Series): The returns of the benchmark in order to measure the performance of the backtest. The Series must have a `DatetimeIndex` with freq=B.
        """
        lower_bound = max(
            [
                universe_returns.index[0],
                market_returns.index[0],
                benchmark_returns.index[0],
            ]
        )
        upper_bound = min(
            [
                universe_returns.index[-1],
                market_returns.index[-1],
                benchmark_returns.index[-1],
            ]
        )
        self.__universe_returns = universe_returns.loc[lower_bound:upper_bound]
        self.__market_returns = market_returns.loc[lower_bound:upper_bound]
        self.__benchmark_returns = benchmark_returns.loc[lower_bound:upper_bound]

    def run_backtest(
        self,
        allocation_type: AllocationMethodsEnum,
        rebalance_frequency: RebalanceFrequencyEnum,
        market_regime_model: RegimeDetectionModels,
        regime_frequency: RebalanceFrequencyEnum = RebalanceFrequencyEnum.BI_MONTHLY,
        transaction_cost_by_securities: Optional[Dict[str, float]] = None,
        bullish_leverage_by_securities: Optional[Dict[str, float]] = None,
        bearish_leverage_by_securities: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        print_metrics: bool = True,
        plot_performance: bool = True,
        starting_offset: int = 20,
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        """_summary_

        Args:
        ----
            allocation_type (AllocationMethodsEnum): The allocation method to use. Only Equally weighted here.
            rebalance_frequency (RebalanceFrequencyEnum): The portfolio rebalance frequency usually monthly.
            market_regime_model (RegimeDetectionModels): The market regime detection model to use.
            transaction_cost_by_securities (Dict[str, float]): A dictionary mapping the securities of the universe and their transaction costs
            bullish_leverage_by_securities (Optional[Dict[str, float]], optional): A dictionary mapping the securities of the universe and their leverage during bullish market regime. Defaults to None.
            bearish_leverage_by_securities (Optional[Dict[str, float]], optional): A dictionary mapping the securities of the universe and their leverage during bearish market regime. Defaults to None.
            verbose (bool, optional): Print rebalance dates... Defaults to True.
            print_metrics (bool, optional): Print the metrics of the strategy. Defaults to True.
            plot_performance (bool, optional): Plot several Chart showing the performances of the strategy. Defaults to True.
            starting_offset (int, optional): _description_. Defaults to 20.

        Returns:
        ----
            Tuple[Union[pd.Series, pd.DataFrame], ...]: Return a tuple of DataFrame/Series respectively : The returns of the strategy and the bench ptf_and_bench (Series), the historical daily weights of the portfolio ptf_weights_df (DataFrame), The regime a the beta at each detection date ptf_regime_beta_df (DataFrame), all risk/perf metrics of the strategy ptf_metrics_df (DataFrame)
        """
        assert starting_offset >= 0, "Error, provide a positive starting offset."
        # Handle null leverages and fees
        if transaction_cost_by_securities is None:
            transaction_cost_by_securities = {
                k: 0.0 for k in self.__universe_returns.columns
            }
        if bearish_leverage_by_securities is None:
            bearish_leverage_by_securities = {
                k: 1 for k in self.__universe_returns.columns
            }
        if bullish_leverage_by_securities is None:
            bullish_leverage_by_securities = {
                k: 1 for k in self.__universe_returns.columns
            }

        # List to store the returns and weights at each iteration (i.e. Days)
        regimes_histo: List[Dict[str, Union[float, int, pd.Timestamp, datetime]]] = []
        returns_histo = []
        weights_histo: List[Dict[str, float]] = []

        # Get all rebalance dates during the backtest
        REBALANCE_DATES = get_rebalance_dates(
            start_date=self.__universe_returns.index[0],
            end_date=self.__universe_returns.index[-1],
            frequency=rebalance_frequency,
        )
        # Get all detection dates for running the regime inference
        DETECTION_DATES = get_rebalance_dates(
            start_date=self.__universe_returns.index[0],
            end_date=self.__universe_returns.index[-1],
            frequency=regime_frequency,
        )

        first_rebalance = False  # Create a portfolio at the first date of the backtest
        leverage_to_use = bullish_leverage_by_securities
        for index, row in tqdm(
            self.__universe_returns.iloc[starting_offset:].iterrows(),
            desc="Backtesting portfolio...",
            total=self.__universe_returns.shape[0],
            leave=False,
        ):
            if index in REBALANCE_DATES or first_rebalance is False:
                # Detect market regime on the market variable provided it has to be returns
                REGIMES = detect_market_regime(
                    self.__market_returns.loc[:index].to_numpy().reshape(-1, 1),
                    market_regime_detection_algorithm=market_regime_model,
                    scale_data=True,
                    scaler_type="robust",
                )
                next_beta = 1
                # next_beta = predict_next_beta_and_alpha(
                #     market_returns=self.__market_returns.loc[:index]
                #     # .resample("1W-FRI")
                #     # .last()
                #     .to_numpy(),
                #     asset_returns=self.__benchmark_returns.loc[:index]
                #     # .resample("1W-FRI")
                #     # .last()
                #     .to_numpy(),
                # )[-1]
                regimes_histo.append(
                    {"Date": index, "Regime": REGIMES[-1], "next_beta": next_beta}
                )
                if REGIMES[-1] == 1:  # Bearish market
                    leverage_to_use = bearish_leverage_by_securities
                else:
                    leverage_to_use = bullish_leverage_by_securities
                # assets = Universe.get_universe_securities()
                first_rebalance = True
                if verbose:
                    print(f"Rebalancing the portfolio on {index}...")
                weights = allocate_assets(
                    row.index.to_list(), 0.35 if REGIMES[-1] == 1 else 0.1
                )
                # ALLOCATION_DICT[allocation_type](
                #     assets,
                #     self.__universe_returns[assets].loc[:index]
                #     # self.__universe_returns.columns, self.__universe_returns.loc[:index]
                # )
                # Row returns with applied fees
                returns_with_applied_fees = []
                for ind, value in row.loc[list(weights.keys())].items():
                    returns_with_applied_fees.append(
                        (value - transaction_cost_by_securities.get(str(ind), 0.0))
                        * leverage_to_use.get(str(ind), 1)
                    )
                returns = np.array(returns_with_applied_fees)
            elif index in DETECTION_DATES:
                # Detect market regime on the market variable provided it has to be returns
                REGIMES = detect_market_regime(
                    self.__market_returns.loc[:index].to_numpy().reshape(-1, 1),
                    market_regime_detection_algorithm=market_regime_model,
                    scale_data=True,
                    scaler_type="robust",
                )
                next_beta = 1
                # next_beta = predict_next_beta_and_alpha(
                #     market_returns=self.__market_returns.loc[:index]
                #     # .resample("1W-FRI")
                #     # .last()
                #     .to_numpy(),
                #     asset_returns=self.__benchmark_returns.loc[:index]
                #     # .resample("1W-FRI")
                #     # .last()
                #     .to_numpy(),
                # )[-1]
                regimes_histo.append(
                    {"Date": index, "Regime": REGIMES[-1], "next_beta": next_beta}
                )
                if regimes_histo[-2].get("Regime") == regimes_histo[-1].get("Regime"):
                    # Row returns
                    returns = row.loc[list(weights.keys())].to_numpy()
                else:
                    if REGIMES[-1] == 1:  # Bearish market
                        leverage_to_use = bearish_leverage_by_securities
                    else:
                        leverage_to_use = bullish_leverage_by_securities
                    # assets = Universe.get_universe_securities()
                    first_rebalance = True
                    if verbose:
                        print(f"Rebalancing the portfolio on {index}...")
                    weights = allocate_assets(
                        row.index.to_list(), 0.35 if REGIMES[-1] == 1 else 0.1
                    )
                    # weights = ALLOCATION_DICT[allocation_type](
                    #     assets,
                    #     self.__universe_returns[assets].loc[:index]
                    #     # self.__universe_returns.columns, self.__universe_returns.loc[:index]
                    # )
                    # Row returns with applied fees
                    returns_with_applied_fees = []
                    for ind, value in row.loc[list(weights.keys())].items():
                        returns_with_applied_fees.append(
                            (value - transaction_cost_by_securities.get(str(ind), 0.0))
                            * leverage_to_use.get(str(ind), 1)
                        )
                    returns = np.array(returns_with_applied_fees)
            else:
                # Row returns
                # ret_to_use= row.loc[list(weights.keys())]
                returns = row.loc[list(weights.keys())].to_numpy()
                # returns_with_applied_fees = []
                # for ind, value in row.loc[list(weights.keys())].items():
                #     returns_with_applied_fees.append(
                #         (value) * leverage_to_use.get(str(ind), 1)
                #     )
                # returns = np.array(returns_with_applied_fees)
            weights = {
                sec: w * leverage_to_use.get(sec, 1) for sec, w in weights.items()
            }
            # Append the current weights to the list
            weights_histo.append(weights)
            # Create numpy weights for matrix operations
            weights_np = np.array(list(weights.values()))
            # Append the current return to the list
            returns_histo.append((returns @ weights_np))

            # Compute the weight drift due to assets price fluctuations
            weights = compute_weights_drift(
                list(weights.keys()),
                weights_np,
                (returns * weights_np),
            )

        # The returns of the ptf
        ptf_returns = pd.Series(
            returns_histo,
            index=self.__universe_returns.iloc[starting_offset:].index,
            dtype=float,
            name="strategy_returns",
        )

        # Construct dataframe with the returns, the perf, and the drawdown for the plots.
        self.__benchmark_returns.name = "returns"
        ptf_and_bench = pd.merge(
            ptf_returns, self.__benchmark_returns, left_index=True, right_index=True
        )
        ptf_and_bench["cum_returns"] = (ptf_and_bench["returns"] + 1).cumprod()
        ptf_and_bench["strategy_cum_returns"] = (
            ptf_and_bench["strategy_returns"] + 1
        ).cumprod()
        ptf_and_bench["drawdown"] = drawdown(ptf_and_bench["returns"])
        ptf_and_bench["strategy_drawdown"] = drawdown(ptf_and_bench["strategy_returns"])

        # The weights of the ptf
        ptf_weights_df = pd.DataFrame(
            weights_histo,
            index=self.__universe_returns.iloc[starting_offset:].index,
            dtype=float,
        ).fillna(0)

        # Store the beta and regime in a dataframe for later analysis
        ptf_regime_beta_df = pd.DataFrame(
            regimes_histo,
        ).set_index("Date")

        if print_metrics is True:
            ptf_metrics_df = print_portfolio_strategy_report(
                ptf_and_bench["strategy_returns"],
                ptf_and_bench["returns"],
            )
        if plot_performance is True:
            plot_from_trade_df(
                ptf_and_bench,
                ptf_weights_df,
                ptf_regime_beta_df,
            )
        if print_metrics is True:
            return ptf_and_bench, ptf_weights_df, ptf_regime_beta_df, ptf_metrics_df
        return ptf_and_bench, ptf_weights_df, ptf_regime_beta_df


def allocate_assets(list_of_assets: List[str], ester_weight: float) -> Dict[str, float]:
    assert (
        "ESTR_ETF" in list_of_assets
    ), "Error, provide a valid list of assets, it must contain the ESTR_ETF columns which is the monetary deposit rate."
    assert 0 <= ester_weight <= 0.6, "Error equity allocation cannot be lower then 40 %"
    list_of_assets.remove("ESTR_ETF")
    weights = {
        security: weight_base_1 * (1 - ester_weight)
        for security, weight_base_1 in Allocation.equal_weighted_allocation(
            list_of_assets
        ).items()
    }
    weights["ESTR_ETF"] = ester_weight
    return weights
