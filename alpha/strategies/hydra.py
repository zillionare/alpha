from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy


class HydraStrategy(BaseXGBoostStrategy):
    """
    The Hydra Strategy is based on xgboost, but use several hybrid method to extract features:

    1. vecter matching

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseXGBoostStrategy class

        :param args:
        :param kwargs:
        """

        BaseXGBoostStrategy.__init__(self, *args, **kwargs)
