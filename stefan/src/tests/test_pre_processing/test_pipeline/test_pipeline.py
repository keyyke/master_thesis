import os
import unittest

import pandas as pd
from unittest.mock import patch

from src.pre_processing.flags.flag import Flag
from src.pre_processing.interfaces.executor_interface import ExecutorInterface
from src.pre_processing.pipeline.pipeline import Pipeline
from src.pre_processing.tasks.task import Task


class Test_Pipeline(unittest.TestCase):
    class myExecutor(ExecutorInterface):
        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            dataToProcess['A'] = 2 * dataToProcess['A']
            return dataToProcess

    def setUp(self):
        self.flag = Flag("Flag1")
        self.task = Task('TestTask', 1, executor=self.myExecutor(), flags=[self.flag])

        self.pipeline = Pipeline('TestPath')

    @patch.object(pd, 'read_csv')
    def test_load_data(self, mock):
        mock.return_value = pd.DataFrame([[2, -1], [1, 3]], columns=['A', 'B'])
        testData = self.pipeline.loadData().getData()
        self.assertEqual([[2, -1], [1, 3]], testData.to_numpy().tolist())
        self.assertEqual(['A', 'B'], list(testData.columns))

    @patch.object(pd, 'read_csv')
    def test_getter_and_setter(self, mock):
        mock.return_value = pd.DataFrame([[2, -1], [1, 3]], columns=['A', 'B'])

        self.assertEqual(self.pipeline.getPath(), 'TestPath')
        self.pipeline.setPath("test").setTask(self.task)

        self.assertEqual("test", self.pipeline.getPath())
        self.assertIsNotNone(self.pipeline.getTask())

        with self.assertRaises(AssertionError):
            self.pipeline.setTask("test")

    @patch.object(pd, 'read_csv')
    def test_run(self, mock):
        mock.return_value = pd.DataFrame([[2, -1], [1, 3]], columns=['A', 'B'])
        self.pipeline.loadData().setTask(self.task)

        testData = self.pipeline.run()
        self.assertEqual([[2, -1], [1, 3]], self.pipeline.getData().to_numpy().tolist())
        self.assertEqual([[4, -1], [2, 3]], testData.to_numpy().tolist())

    @patch.object(pd, 'read_csv')
    def test_run_with_flags(self, mock):
        mock.return_value = pd.DataFrame([[2, -1], [1, 3]], columns=['A', 'B'])
        self.pipeline.loadData().setTask(self.task)

        testData = self.pipeline.run(flag=self.flag)
        self.assertEqual([[2, -1], [1, 3]], self.pipeline.getData().to_numpy().tolist())
        self.assertEqual([[4, -1], [2, 3]], testData.to_numpy().tolist())

        self.pipeline.loadData().setTask(self.task)

        testData = self.pipeline.run(flag=Flag("Flag123"))
        self.assertEqual([[2, -1], [1, 3]], testData.to_numpy().tolist())

    def test_error_before_loading_data(self):
        with self.assertRaises(Exception):
            self.pipeline.run()
        with self.assertRaises(Exception):
            self.pipeline.getData()
