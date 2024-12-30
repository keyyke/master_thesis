from __future__ import annotations

import pandas as pd

from src.pre_processing.flags.abstract_flag import AbstractFlag
from src.pre_processing.flags.base_flag import BaseFlag
from src.pre_processing.interfaces.task_group_interface import TaskGroupInterface
from src.pre_processing.tasks.abstract_task import AbstractTask
from src.pre_processing.custom_queues.custom_priority_queue import CustomPriorityQueue


class TaskGroup(AbstractTask, TaskGroupInterface):
    def __init__(self, name, priority: int = None, flags: list[AbstractFlag] | AbstractFlag = BaseFlag(),
                 inplace: bool = True, idempotency: bool = False):
        super().__init__(name, priority, flags, inplace)

        self.__taskQueue = CustomPriorityQueue()
        self.__idempotency = idempotency

    def add(self, task: AbstractTask) -> TaskGroup:
        assert isinstance(task, AbstractTask), 'Object not instance of AbstractTask'

        if self.__idempotency:
            if str(task) in self.__taskQueue:
                del self.__taskQueue[str(task)]

        self.__taskQueue.push(task)
        return self

    def getIdempotency(self) -> bool:
        return self.__idempotency

    def addAll(self, tasks: list[AbstractTask]) -> TaskGroup:
        for task in tasks:
            self.add(task)
        return self

    def getAll(self) -> list[AbstractTask]:
        return self.__taskQueue.getAll()

    def reset(self) -> TaskGroup:
        self.__taskQueue.reset()
        return self

    def size(self) -> int:
        return self.__taskQueue.size()

    def __len__(self) -> int:
        return len(self.__taskQueue)

    def process(self, dataToProcess: pd.DataFrame, flag: AbstractFlag) -> pd.DataFrame:
        for taskFLag in self.getFlags():
            if flag.contains(taskFLag):
                for task in self.__taskQueue:
                    if self.isInplace():
                        dataToProcess = task.process(dataToProcess, flag)
                    else:
                        task.process(dataToProcess.copy(deep=True), flag)
                break

        return dataToProcess

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        for task in self.__taskQueue:
            if self.isInplace():
                dataToProcess = task.execute(dataToProcess)
            else:
                task.execute(dataToProcess.copy(deep=True))

        return dataToProcess

    def _print(self, ident: int = 0, priority: bool = False, flags: bool = False):
        super()._print(ident=ident, flags=flags, priority=priority)

        for task in self.__taskQueue:
            task._print(ident=ident + 1, flags=flags, priority=priority)

    def __getitem__(self, subscript):
        return self.__taskQueue[subscript]

    def __delitem__(self, subscript):
        del self.__taskQueue[subscript]

    def copy(self) -> TaskGroup:
        copyTaskGroup = TaskGroup(self.getName(), self.getPriority(), self.getFlags(), self.isInplace(),
                                  self.getIdempotency())
        allTasks = self.getAll()

        for task in allTasks:
            copyTaskGroup.add(task.copy())

        return copyTaskGroup
