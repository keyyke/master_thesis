import unittest

from src.plotting.plotter import Plotter
from src.pre_processing.tasks.task import Task
from src.plotting.plot_factory import PlotPipelineFactory


class Test_PlotPipelineFactory(unittest.TestCase):
    def test_modes(self):
        pipeline = PlotPipelineFactory(titleSuffix='', mode='analysis')

        self.assertEqual(len(pipeline), 24)

        pipeline = PlotPipelineFactory(titleSuffix='', mode='modeling')

        self.assertEqual(len(pipeline), 20)

        pipeline = PlotPipelineFactory(titleSuffix='', mode='inference')

        self.assertEqual(len(pipeline), 18)

        with self.assertRaises(AssertionError):
            pipeline = PlotPipelineFactory(titleSuffix='', mode='hello')

    def test_tasks(self):
        pipeline = PlotPipelineFactory(titleSuffix='TEST', savePath='PATH', dpi=100, figSize=(3, 4),
                                       show=False, columns={'HI': 'HEllO'}, labelColumn='My_Label',
                                       mode='analysis')

        for task in pipeline[:-1]:
            if isinstance(task, Task):
                executor = task.getExecutor()

                self.assertTrue(isinstance(executor, Plotter))

                self.assertEqual(executor.title[-4:], 'TEST')
                self.assertEqual(executor.dpi, 100)
                self.assertEqual(executor.savePath, 'PATH')
                self.assertEqual(executor.figSize, (3, 4))
                self.assertEqual(executor.show, False)
                self.assertEqual(executor.noneValue, 'None')
                self.assertEqual(executor.otherValue, 'Other')

            else:
                raise Exception('invalid type')
