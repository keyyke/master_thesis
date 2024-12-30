from abc import abstractmethod, ABC
import pandas as pd

from src.pre_processing.flags.abstract_flag import AbstractFlag


class TaskInterface(ABC):

    @abstractmethod
    def getPriority(self) -> int | None:
        pass

    @abstractmethod
    def print(self, priority: bool = False, flags: bool = False):
        pass

    @abstractmethod
    def process(self, dataToProcess: pd.DataFrame, flag: AbstractFlag) -> pd.DataFrame:
        pass

    @abstractmethod
    def getFlags(self) -> list[AbstractFlag]:
        pass

    @abstractmethod
    def isInplace(self) -> bool:
        pass

    @abstractmethod
    def getName(self) -> str:
        pass



