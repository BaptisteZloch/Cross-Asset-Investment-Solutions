import pandas as pd


class Benchmark:
    _PATH = "../data/benchmark.csv"

    @staticmethod
    def get_benchmark_price_data() -> pd.DataFrame:
        return pd.read_csv(
            Benchmark._PATH,
            sep=",",
            date_format="%Y-%m-%d",
            parse_dates=True,
            index_col="Date",
        )

    @staticmethod
    def get_benchmark_historical_track(rebase_at:int=100) -> pd.DataFrame:
        Benchmark.get_benchmark_price_data().pct_change().dropna()
        return pd.read_csv(
            Benchmark._PATH,
            sep=",",
            date_format="%Y-%m-%d",
            parse_dates=True,
            index_col="Date",
        )
