from abc import ABC, abstractmethod

import pandas as pd


class ExecutorInterface(ABC):
    @abstractmethod
    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        pass
