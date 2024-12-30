import pandas as pd

from src.pre_processing.tasks.abstract_task import AbstractTask
from src.pre_processing.tasks.task import Task
from src.pre_processing.flags.abstract_flag import AbstractFlag


class Pipeline:
    def __init__(self, filePath: str = None, task: AbstractTask = Task("Default")):
        self.__path = filePath
        self.__data = None
        self.__task = task

    def loadData(self):
        self.__data = pd.read_csv(self.__path, encoding='cp1252')
        return self

    def getData(self) -> pd.DataFrame:
        self.__checkData()
        return self.__data

    def getPath(self) -> str:
        return self.__path

    def setPath(self, path):
        self.__path = path
        return self

    def setTask(self, task: AbstractTask):
        assert isinstance(task, AbstractTask)
        self.__task = task
        return self

    def getTask(self) -> AbstractTask:
        return self.__task

    def __checkData(self):
        if self.__data is None:
            raise Exception('no data was loaded')

    def run(self, flag: AbstractFlag = None) -> pd.DataFrame:
        self.__checkData()

        if flag is None:
            return self.__task.execute(self.__data.copy(deep=True))
        else:
            return self.__task.process(self.__data.copy(deep=True), flag)
