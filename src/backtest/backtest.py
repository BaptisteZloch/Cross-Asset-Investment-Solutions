from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from backtest.metrics import drawdown
from backtest.reports import plot_from_trade_df, print_portfolio_strategy_report
from data.universe import Universe
from portfolio_management.allocation import Allocation
from portfolio_management.beta_estimation import beta_convergence
from portfolio_management.market_regime import detect_market_regime
from utility.types import (
    AllocationMethodsEnum,
    RebalanceFrequencyEnum,
    RegimeDetectionModels,
)

from utility.utils import (
    compute_weights_drift,
    get_rebalance_dates,
)


class Backtester:
    def __init__(
        self,
        universe_obj: Universe,
        benchmark_returns: pd.Series,
    ) -> None:
        """Constructor for the backtester class

        Args:
        ----
            universe_obj (Universe): The class object representing the universe (securities, dataframe for prices and returns...). Use the Universe class.
            benchmark_returns (pd.Series): The returns of the benchmark in order to measure the performance of the backtest. The Series must have a `DatetimeIndex` with freq=B.
        """
        self.__universe_obj = universe_obj
        universe_returns = self.__universe_obj.get_universe_returns()
        lower_bound = max(
            [
                universe_returns.index[0],
                benchmark_returns.index[0],
            ]
        )
        upper_bound = min(
            [
                universe_returns.index[-1],
                benchmark_returns.index[-1],
            ]
        )
        self.__universe_returns = universe_returns.loc[lower_bound:upper_bound]
        self.__market_returns = self.__universe_obj.market_returns.loc[
            lower_bound:upper_bound
        ]
        self.__benchmark_returns = benchmark_returns.loc[lower_bound:upper_bound]

    def run_backtest(
        self,
        rebalance_frequency: RebalanceFrequencyEnum,
        market_regime_model: RegimeDetectionModels,
        regime_frequency: RebalanceFrequencyEnum = RebalanceFrequencyEnum.BI_MONTHLY,
        bullish_beta: float = 1.2,
        bearish_beta: float = 0.9,
        bullish_lambda_convergence: int = 50,
        bearish_lambda_convergence: int = 50,
        verbose: bool = True,
        print_metrics: bool = True,
        plot_performance: bool = True,
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        """_summary_

        Args:
            rebalance_frequency (RebalanceFrequencyEnum): The portfolio rebalance frequency usually monthly.
            market_regime_model (RegimeDetectionModels): The market regime detection model to use.
            regime_frequency (RebalanceFrequencyEnum, optional): _description_. Defaults to RebalanceFrequencyEnum.BI_MONTHLY.
            bullish_beta (float, optional): _description_. Defaults to 1.2.
            bearish_beta (float, optional): _description_. Defaults to 0.9.
            verbose (bool, optional): Print rebalance dates... Defaults to True.
            print_metrics (bool, optional): Print the metrics of the strategy. Defaults to True.
            plot_performance (bool, optional): Plot several Chart showing the performances of the strategy. Defaults to True.
            20 (int, optional): _description_. Defaults to 20.

        Returns:
            Tuple[Union[pd.Series, pd.DataFrame], ...]: Return a tuple of DataFrame/Series respectively : The returns of the strategy and the bench ptf_and_bench (Series), the historical daily weights of the portfolio ptf_weights_df (DataFrame), The regime a the beta at each detection date ptf_regime_beta_df (DataFrame), all risk/perf metrics of the strategy ptf_metrics_df (DataFrame)
        """
        # Handle fees (no fees here)
        transaction_cost_by_securities = {
            k: 0.0 for k in self.__universe_returns.columns
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

        # initial values used in the backtest

        first_rebalance = False  # Create a portfolio at the first date of the backtest
        target_beta = bullish_beta  # Long term beta the portfolio needs to converge to
        initial_period_beta = 1  # Store the initial beta (expo) value, this value will be updated at each regime change it's used in the beta convergence function.
        new_beta = bullish_beta  # The current beta
        days_in_same_regime = 0  # Count the number of days in the same regime (used in the converge beta functon)
        leverage_to_use = {
            k: 1 if k not in ["Px fut SX5E", "Px fut sp500","Px fut nasdaq"] else new_beta
            for k in self.__universe_returns.columns
        }
        for index, row in tqdm(
            self.__universe_returns.iloc[20:].iterrows(),
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
                initial_period_beta = new_beta
                days_in_same_regime = 0
                target_beta = bearish_beta if REGIMES[-1] == 1 else bullish_beta

                # assets = Universe.get_universe_securities()
                first_rebalance = True
                if verbose:
                    print(f"Rebalancing the portfolio on {index}...")
                weights = allocate_assets(
                    self.__universe_obj.defensive_securities
                    if REGIMES[-1] == 1
                    else self.__universe_obj.offensive_securities,
                    0.2
                    # row.index.to_list(), 0.35 if REGIMES[-1] == 1 else 0.1
                )
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

                if regimes_histo[-2].get("Regime") == regimes_histo[-1].get("Regime"):
                    # Row returns
                    returns = row.loc[list(weights.keys())].to_numpy()
                else:
                    initial_period_beta = new_beta
                    days_in_same_regime = 0
                    target_beta = bearish_beta if REGIMES[-1] == 1 else bullish_beta
                    first_rebalance = True
                    if verbose:
                        print(f"Rebalancing the portfolio on {index}...")
                    weights = allocate_assets(
                        self.__universe_obj.defensive_securities
                        if REGIMES[-1] == 1
                        else self.__universe_obj.offensive_securities,
                        0.2
                        # row.index.to_list(), 0.35 if REGIMES[-1] == 1 else 0.1
                    )
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
            days_in_same_regime += 1
            new_beta = beta_convergence(
                current_value=days_in_same_regime,
                initial_value=initial_period_beta,
                long_term_value=target_beta,
                smoothing_lambda=bullish_lambda_convergence
                if REGIMES[-1] == 0
                else bearish_lambda_convergence,
            )
            regimes_histo.append(
                {"Date": index, "Regime": REGIMES[-1], "next_beta": new_beta}
            )
            # Only leverage the futures columns
            leverage_to_use = {
                k: 1 if k not in ["Px fut SX5E", "Px fut sp500","Px fut nasdaq"] else new_beta
                for k in self.__universe_returns.columns
            }
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
            index=self.__universe_returns.iloc[20:].index,
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
            index=self.__universe_returns.iloc[20:].index,
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
