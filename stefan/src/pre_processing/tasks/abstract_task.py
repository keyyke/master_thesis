from __future__ import annotations
from abc import abstractmethod, ABC

from src.pre_processing.flags.abstract_flag import AbstractFlag
from src.pre_processing.flags.base_flag import BaseFlag
from src.pre_processing.interfaces.executor_interface import ExecutorInterface
from src.pre_processing.interfaces.task_interface import TaskInterface


class AbstractTask(TaskInterface, ExecutorInterface, ABC):
    def __init__(self, name: str, priority: int = None, flags: list[AbstractFlag] | AbstractFlag = BaseFlag(),
                 inplace: bool = True):

        self.__name = name
        self.__priority = priority
        self.__flags = None
        self.__inplace = inplace
        self.__setFlags(flags)

    def isInplace(self) -> bool:
        return self.__inplace

    def __setFlags(self, flags: list[AbstractFlag] | AbstractFlag):
        if isinstance(flags, AbstractFlag):
            self.__flags = [flags]
        else:
            self.__flags = flags

    def getFlags(self) -> list[AbstractFlag]:
        return self.__flags

    def getPriority(self) -> int | None:
        return self.__priority

    def getName(self) -> str:
        return self.__name

    def __lt__(self, other: AbstractTask) -> bool:
        if self.getPriority() is None and other.getPriority() is not None:
            return False
        elif (self.getPriority() is not None) and (other.getPriority() is None):
            return True
        elif self.getPriority() is None and other.getPriority() is None:
            return False
        else:
            return self.getPriority() < other.getPriority()

    def __gt__(self, other: AbstractTask) -> bool:
        return other < self

    def __str__(self) -> str:
        return self.getName()

    def print(self, priority: bool = False, flags: bool = False):
        self._print(ident=0, priority=priority, flags=flags)

    def _print(self, ident: int = 0, priority: bool = False, flags: bool = False):
        if self.getPriority() is None or not priority:
            print('\t' * ident + self.getName(), end='')
        elif priority:
            print('\t' * ident + str(self.getPriority()) + " " + self.getName(), end='')

        if flags and not self.getFlags() == [BaseFlag()]:
            print(":", end=' ')
            for index, flag in enumerate(self.getFlags()):
                if index < len(self.getFlags()) - 1:
                    print(flag.getName(), end=', ')
                else:
                    print(flag.getName(), end='')

        print("\n", end='')

    @abstractmethod
    def copy(self) -> AbstractTask:
        pass
