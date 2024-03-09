from enum import StrEnum


class RebalanceFrequencyEnum(StrEnum):
    DAILY = "1B"
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


class AllocationMethodsEnum(StrEnum):
    EQUALLY_WEIGHTED = "EQUALLY_WEIGHTED"
    # MAX_SHARPE = "MAX_SHARPE"

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls):
        return list(map(lambda c: c.name, cls))
