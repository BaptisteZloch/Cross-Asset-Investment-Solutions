from enum import auto
from strenum import StrEnum


class RebalanceFrequencyEnum(StrEnum):
    DAILY = "1B"
    WEEKLY = "1W-FRI"
    BI_MONTHLY = "SMS"
    MONTH_END = "1BME"
    MONTH_START = "BMS"
    QUARTER_END = "BQE"
    QUARTER_START = "BQS"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))


class RegimeDetectionModels(StrEnum):
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    HIDDEN_MARKOV_MODEL = "hmm"
    KMEANS = "kmeans"
    BISECTING_KMEANS = "bisecting_kmeans"
    JUMP_MODEL = "jump_model"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))


class AllocationMethodsEnum(StrEnum):
    EQUALLY_WEIGHTED = "EQUALLY_WEIGHTED"
    # MAX_SHARPE = "MAX_SHARPE"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))
