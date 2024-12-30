import io
import sys
import unittest

import pandas as pd

from src.pre_processing.flags.flag import Flag
from src.pre_processing.interfaces.executor_interface import ExecutorInterface
from src.pre_processing.tasks.task import Task
from src.pre_processing.tasks.task_group import TaskGroup


class Test_TaskGroup_With_Flags(unittest.TestCase):
    class simpleExecutor(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = 2 * dataToProcess['A']
            return dataToProcess

    class simpleExecutor2(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = dataToProcess['A'] - 1
            return dataToProcess

    def setUp(self, inplace = True):
        self.flag1 = Flag("Flag1")
        self.flag2 = Flag("Flag2")
        self.flag3 = Flag("Flag3", parents=[self.flag1, self.flag2])
        self.flag4 = Flag("Flag4")

        nestedTaskGroup1 = TaskGroup("NestedGroup1", 2, flags=self.flag1)

        nestedTaskGroup2 = TaskGroup("NestedGroup2", 3, flags=[self.flag2])

        task1 = Task("Task1", executor=self.simpleExecutor(), priority=1, flags=[self.flag1, self.flag2])
        task2 = Task("Task2", executor=self.simpleExecutor2())
        task3 = Task("Task3", executor=self.simpleExecutor(), priority=2, flags=[self.flag3, self.flag4])

        nestedTaskGroup1.addAll([task1, task2])
        nestedTaskGroup2.addAll([task2, task3])

        self.taskGroup = TaskGroup('TaskGroup', inplace=inplace)

        self.taskGroup.addAll([nestedTaskGroup1, nestedTaskGroup2, task1, task2])

        self.data = pd.DataFrame([[3, 1]], columns=['A', 'B'])


    def test_print(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.taskGroup.print(flags=True)

        self.assertEqual('TaskGroup\n' +
                         '\t' + 'Task1: Flag1, Flag2\n' +
                         '\t' + 'NestedGroup1: Flag1\n' +
                         '\t' + '\t' + 'Task1: Flag1, Flag2\n' +
                         '\t' + '\t' + 'Task2\n' +
                         '\t' + 'NestedGroup2: Flag2\n' +
                         '\t' + '\t' + 'Task3: Flag3, Flag4\n' +
                         '\t' + '\t' + 'Task2\n' +
                         '\t' + 'Task2\n',
                         capturedOutput.getvalue())

    def test_flag1(self):
        testData = self.taskGroup.process(self.data.copy() , flag=self.flag1)
        self.assertEqual([[10, 1]], testData.to_numpy().tolist())

        testData = self.taskGroup.execute(self.data)
        self.assertEqual([[20, 1]], testData.to_numpy().tolist())

    def test_flag2(self):
        testData = self.taskGroup.process(self.data , flag=self.flag2)

        self.assertEqual([[4, 1]], testData.to_numpy().tolist())

    def test_flag3(self):
        testData = self.taskGroup.process(self.data , flag=self.flag3)

        self.assertEqual([[20, 1]], testData.to_numpy().tolist())

    def test_flag4(self):
        testData = self.taskGroup.process(self.data , flag=self.flag4)

        self.assertEqual([[2, 1]], testData.to_numpy().tolist())

    def test_flag5(self):
        testData = self.taskGroup.process(pd.DataFrame([[3, 1]], columns=['A', 'B']), flag=Flag("Flag5"))

        self.assertEqual([[2, 1]], testData.to_numpy().tolist())

    def test_inplace_false(self):
        self.setUp(inplace=False)

        testData = self.taskGroup.process(pd.DataFrame([[3, 1]], columns=['A', 'B']), flag=self.flag1)
        self.assertEqual([[3, 1]], testData.to_numpy().tolist())

