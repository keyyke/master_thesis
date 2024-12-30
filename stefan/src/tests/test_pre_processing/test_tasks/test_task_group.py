import io
import sys
import unittest

import pandas as pd

from src.pre_processing.interfaces.executor_interface import ExecutorInterface
from src.pre_processing.tasks.simple_executor import SimpleExecutor
from src.pre_processing.tasks.task import Task
from src.pre_processing.tasks.task_group import TaskGroup


class Test_TaskGroup(unittest.TestCase):
    class simpleExecutor1(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = 2 * dataToProcess['A']
            return dataToProcess

    class simpleExecutor2(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = dataToProcess['A'] - 1
            return dataToProcess

    def setUp(self):
        self.task = Task('TestTask', 1, executor=self.simpleExecutor1())

        self.task2 = Task('TestTask2', 3, executor=self.simpleExecutor2())
        self.task3 = Task('TestTask3', executor=self.simpleExecutor2())
        self.data = pd.DataFrame([[1, 3]], columns=['A', 'B'])
        self.taskGroup = TaskGroup('TaskGroup')

    def test_add(self):
        self.taskGroup.add(self.task2)
        self.taskGroup.add(self.task)
        self.taskGroup.addAll([self.task3, self.task2])

        self.assertEqual(4, self.taskGroup.size())
        self.assertEqual(len(self.taskGroup), 4)

        self.assertEqual(self.taskGroup.getAll(), [self.task, self.task2, self.task2, self.task3])

        testData = self.data.copy(deep=True)
        testData = self.taskGroup.execute(testData)
        self.assertEqual([[-1, 3]], testData.to_numpy().tolist())

        self.assertEqual(4, self.taskGroup.size())

        testData = self.data.copy(deep=True)
        testData = self.taskGroup.execute(self.data)
        self.assertEqual([[-1, 3]], testData.to_numpy().tolist())

    def test_reset(self):
        self.taskGroup.add(self.task2)
        self.taskGroup.add(self.task)

        self.assertEqual(2, self.taskGroup.size())
        self.taskGroup.reset()
        self.assertEqual(0, self.taskGroup.size())

        testData = self.taskGroup.execute(self.data)
        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

    def test_false_type_added(self):
        with self.assertRaises(AssertionError):
            self.taskGroup.add(SimpleExecutor())

    def test_nested_TaskGroup(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task3, self.task, self.task2])
        self.taskGroup.addAll([self.task3, self.task2, nestedTaskGroup, self.task3, self.task, self.task])

        testData = self.data.copy(deep=True)
        testData = self.taskGroup.execute(testData)
        self.assertEqual([[3, 3]], testData.to_numpy().tolist())

        self.assertEqual(6, self.taskGroup.size())

    def test_print(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task3, self.task, self.task2])
        self.taskGroup.addAll([self.task3, self.task2, nestedTaskGroup, self.task3, self.task, self.task])

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        self.taskGroup.print()

        self.assertEqual('TaskGroup\n' +
                         '\t' + 'TestTask\n' +
                         '\t' + 'TestTask\n' +
                         '\t' + 'NestedGroup\n' +
                         '\t' + '\t' + 'TestTask\n' +
                         '\t' + '\t' + 'TestTask2\n' +
                         '\t' + '\t' + 'TestTask3\n' +
                         '\t' + 'TestTask2\n' +
                         '\t' + 'TestTask3\n' +
                         '\t' + 'TestTask3\n',
                         capturedOutput.getvalue())

    def test_print_priority(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task3, self.task, self.task2])
        self.taskGroup.addAll([self.task3, self.task2, nestedTaskGroup, self.task3, self.task, self.task])

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        self.taskGroup.print(priority=True)

        self.assertEqual('TaskGroup\n' +
                         '\t' + '1 TestTask\n' +
                         '\t' + '1 TestTask\n' +
                         '\t' + '2 NestedGroup\n' +
                         '\t' + '\t' + '1 TestTask\n' +
                         '\t' + '\t' + '3 TestTask2\n' +
                         '\t' + '\t' + 'TestTask3\n' +
                         '\t' + '3 TestTask2\n' +
                         '\t' + 'TestTask3\n' +
                         '\t' + 'TestTask3\n',
                         capturedOutput.getvalue())

    def test_subscription(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task, self.task2])

        self.taskGroup.addAll([self.task3, self.task2, nestedTaskGroup])
        self.taskGroup.addAll([self.task3, self.task2, self.task])

        self.assertTrue(self.taskGroup[1] is nestedTaskGroup)
        self.assertTrue(self.taskGroup[-3] is self.task2)
        self.assertTrue(self.taskGroup[1][0] is self.task)
        self.assertEqual([nestedTaskGroup, self.task2], self.taskGroup[1:-1:2])

        self.taskGroup["NestedGroup"].add(self.task3)
        self.assertEqual(nestedTaskGroup.size(), 3)

        self.assertTrue(self.taskGroup["NestedGroup"] is nestedTaskGroup)

    def test_subscript_error(self):
        self.taskGroup.addAll([self.task3, self.task2, self.task])

        with self.assertRaises(Exception):
            element = self.taskGroup['Test']
        with self.assertRaises(Exception):
            element = self.taskGroup[55]
        with self.assertRaises(Exception):
            element = self.taskGroup[4.3]

    def test_idempotency(self):
        myTaskGroup = TaskGroup("TaskGroup", idempotency=True)
        myTaskGroup.addAll([self.task3, self.task, self.task2])

        self.assertEqual(myTaskGroup.size(), 3)
        self.assertEqual(myTaskGroup.getAll(), [self.task, self.task2, self.task3])

        self.assertEqual(myTaskGroup.getIdempotency(), True)

        myTask = Task('TestTask', priority=-5)
        myTaskGroup.add(myTask)

        self.assertEqual(myTaskGroup.size(), 3)
        self.assertEqual(myTaskGroup.getAll(), [myTask, self.task2, self.task3])

        self.assertEqual(myTaskGroup[0].getPriority(), -5)

        myTask2 = Task('TestTask4', priority=100)
        myTaskGroup.add(myTask2)

        self.assertEqual(myTaskGroup.size(), 4)
        self.assertEqual(myTaskGroup.getAll(), [myTask, self.task2, myTask2, self.task3])

    def test_inplace_simple(self):
        newTask = Task('TestTaskNotInplace', priority=2, inplace=False, executor=self.simpleExecutor1())
        self.taskGroup.addAll([self.task3, newTask, self.task, self.task2])

        self.taskGroup.print()

        testData = self.taskGroup.execute(self.data)
        self.assertEqual([[0, 3]], testData.to_numpy().tolist())


    def test_inplace(self):
        taskgroup = TaskGroup('TaskGroupNotInplace', inplace=False)
        taskgroup.addAll([self.task3, self.task, self.task2])

        testData = taskgroup.execute(self.data)
        self.assertEqual([[1, 3]], testData.to_numpy().tolist())

    def test_delete(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task, self.task2])

        self.taskGroup.addAll([self.task3, self.task2, nestedTaskGroup])
        self.taskGroup.addAll([self.task3, self.task2, self.task])

        del self.taskGroup['TestTask2']

        self.assertEqual([self.task, nestedTaskGroup, self.task2, self.task3, self.task3], self.taskGroup.getAll())

        del self.taskGroup[1][1]
        self.assertEqual([self.task], nestedTaskGroup.getAll())

        del self.taskGroup[1:4]

        self.assertEqual([self.task, self.task3], self.taskGroup.getAll())

    def test_copy(self):
        nestedTaskGroup = TaskGroup("NestedGroup", 2)
        nestedTaskGroup.addAll([self.task, self.task2])

        self.taskGroup.addAll([self.task, self.task2, nestedTaskGroup])

        taskGroupCopy = self.taskGroup.copy()
        self.assertEqual(3, len(taskGroupCopy))

        del self.taskGroup[0]
        self.assertEqual(3, len(taskGroupCopy))

        taskCopy = taskGroupCopy[0]

        self.assertFalse(taskCopy is self.task)
        self.assertEqual(type(taskCopy), type(self.task))

        self.assertEqual(taskCopy.getName(), self.task.getName())
        self.assertEqual(taskCopy.getPriority(), self.task.getPriority())
        self.assertEqual(taskCopy.getFlags(), self.task.getFlags())

        self.assertEqual(taskCopy.isInplace(), self.task.isInplace())

        testData = taskCopy.execute(self.data)
        self.assertEqual([[2, 3]], testData.to_numpy().tolist())
