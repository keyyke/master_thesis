import io
import sys
import unittest

import pandas as pd

from src.pre_processing.flags.base_flag import BaseFlag
from src.pre_processing.flags.flag import Flag
from src.pre_processing.interfaces.executor_interface import ExecutorInterface
from src.pre_processing.tasks.task import Task


class Test_AbstractTask(unittest.TestCase):
    class myExecutor(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = 2 * dataToProcess['A']
            return dataToProcess

    def setUp(self):
        self.flag1 = Flag("Flag1")
        self.flag2 = Flag("Flag2")
        self.flag3 = Flag("Flag3", parents=self.flag1)
        self.flag4 = Flag("Flag4", [self.flag1, self.flag2])
        self.flag5 = Flag("Flag")
        self.flag6 = Flag("Flag6", parents=[self.flag3])

        self.task = Task('TestTask', 1, executor=self.myExecutor(), flags=[self.flag2, self.flag1])
        self.data = pd.DataFrame([[1, 3]], columns=['A', 'B'])

    def test_default(self):
        task = Task('Name')
        self.assertEqual('Name', task.getName())
        self.assertIsNone(task.getPriority())
        self.assertTrue(task.getPriority() is None)
        self.assertEqual(task.getFlags(), [BaseFlag()])
        self.assertTrue(task.isInplace())

        testData = task.execute(self.data)
        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

    def test_getter(self):
        task = Task('TestTask2', 4, executor=self.myExecutor(), inplace=False, flags=self.flag1)
        self.assertEqual(4, task.getPriority())
        self.assertFalse(task.isInplace())
        self.assertEqual('TestTask2', task.getName())
        self.assertEqual([self.flag1], task.getFlags())

        self.assertTrue(isinstance(task.getExecutor(), self.myExecutor))

    def test_comparator(self):
        task2 = Task('TestTask2', 2)
        task3 = Task('TestTask3')
        task4 = Task('TestTask4')
        task5 = Task('TestTask5', -1)
        task6 = Task('TestTask6', 2)

        self.assertTrue(self.task < task2)
        self.assertTrue(task2 > self.task)
        self.assertTrue(self.task < task2)
        self.assertFalse(task3 < task2)
        self.assertFalse(task4 < task3)
        self.assertTrue(self.task > task5)
        self.assertFalse(task2 < task6)

    def test_print(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.task.print()
        self.assertEqual('TestTask\n', capturedOutput.getvalue())

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.task.print()
        self.assertEqual('TestTask\n', capturedOutput.getvalue())

    def test_print_flags(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.task.print(flags=True)

        self.assertEqual('TestTask: Flag2, Flag1\n', capturedOutput.getvalue())

    def test_print_priority(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.task.print(priority=True)

        self.assertEqual('1 TestTask\n', capturedOutput.getvalue())

    def test_print_flags_and_priority(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.task.print(flags=True, priority=True)

        self.assertEqual('1 TestTask: Flag2, Flag1\n', capturedOutput.getvalue())

    def test_execute(self):
        testData = self.task.execute(self.data)
        self.assertEqual([[2, 3]], testData.to_numpy().tolist())

    def test_process_data(self):
        testData = self.task.process(self.data, self.flag1)
        self.assertEqual([[2, 3]], testData.to_numpy().tolist())

        testData = self.task.process(self.data, self.flag4)
        self.assertEqual([[4, 3]], testData.to_numpy().tolist())

        testData = self.task.process(self.data, self.flag3)
        self.assertEqual([[8, 3]], testData.to_numpy().tolist())

        testData = self.task.process(self.data, self.flag6)
        self.assertEqual([[16, 3]], testData.to_numpy().tolist())

        testData = self.task.process(self.data, self.flag5)
        self.assertEqual([[16, 3]], testData.to_numpy().tolist())

    def test_inplace(self):
        task = Task("Task", inplace=False, executor=self.myExecutor(), flags=[self.flag2, self.flag1])
        self.assertFalse(task.isInplace())

        testData = task.execute(self.data)
        testData = task.execute(testData)

        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

        testData = task.process(self.data, self.flag4)
        testData = task.process(testData, self.flag4)

        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

        testData = task.process(self.data, self.flag5)
        testData = task.process(testData, self.flag5)

        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

    def test_copy(self):
        taskCopy = self.task.copy()
        self.assertFalse(taskCopy is self.task)
        self.assertEqual(type(taskCopy), type(self.task))

        self.assertEqual(taskCopy.getName(), self.task.getName())
        self.assertEqual(taskCopy.getPriority(), self.task.getPriority())
        self.assertEqual(taskCopy.getFlags(), self.task.getFlags())
        self.assertEqual(taskCopy.isInplace(), self.task.isInplace())

        testData = taskCopy.execute(self.data)
        self.assertEqual([[2, 3]], testData.to_numpy().tolist())
