from abc import abstractmethod, ABC

from src.pre_processing.interfaces.task_interface import TaskInterface


class TaskGroupInterface(TaskInterface, ABC):
    @abstractmethod
    def add(self, task: TaskInterface):
        pass

    @abstractmethod
    def addAll(self, tasks: list[TaskInterface]):
        pass

    @abstractmethod
    def getAll(self) -> list[TaskInterface]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def getIdempotency(self) -> bool:
        pass