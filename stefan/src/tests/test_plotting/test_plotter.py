import unittest

import pandas as pd
import numbers
from src.plotting.plotter import *


class TestPlotter(unittest.TestCase):
    class SimplePlot(Plotter):
        def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            return dataToProcess

        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            return dataToProcess

    def test_plotter_attributes_base(self):
        plotter = self.SimplePlot({'A': 'COl A', 'B': 'COl B'})

        self.assertEqual(plotter.columns, {'A': 'COl A', 'B': 'COl B'})
        self.assertEqual(plotter.figSize, (16, 10))
        self.assertEqual(plotter.dpi, 300)
        self.assertEqual(plotter.title, None)
        self.assertEqual(plotter.xLabel, None)
        self.assertEqual(plotter.yLabel, None)
        self.assertEqual(plotter.savePath, None)
        self.assertEqual(plotter.show, True)
        self.assertEqual(plotter.noneValue, 'None')
        self.assertEqual(plotter.otherValue, 'Other')

    def test_plotter_attributes(self):
        plotter = self.SimplePlot(columns={'A': 'COl A', 'B': 'COl B'},
                                  figSize=(3, 4),
                                  dpi=200,
                                  title='Simple',
                                  xLabel='X',
                                  yLabel='Y',
                                  savePath='PATH',
                                  show=False,
                                  noneValue='NONE',
                                  otherValue='OTHER'
                                  )

        self.assertEqual(plotter.columns, {'A': 'COl A', 'B': 'COl B'})
        self.assertEqual(plotter.figSize, (3, 4))
        self.assertEqual(plotter.dpi, 200)
        self.assertEqual(plotter.title, 'Simple')
        self.assertEqual(plotter.xLabel, 'X')
        self.assertEqual(plotter.yLabel, 'Y')
        self.assertEqual(plotter.savePath, 'PATH')
        self.assertEqual(plotter.show, False)
        self.assertEqual(plotter.noneValue, 'NONE')
        self.assertEqual(plotter.otherValue, 'OTHER')

    def test_getClassLabels(self):
        columns = {'Label': 'LABEL', 'Feature': 'FEATURE'}
        plotter = self.SimplePlot(columns)

        testData = pd.DataFrame([['A', 2], ['C', 3], ['B', -1]], columns=['LABEL', 'FEATURE'])

        labels = plotter.classLabels(testData, 'LABEL')
        self.assertEqual(labels, ['A', 'B', 'C'])

    def test_select_categories(self):
        columns = {'Label': 'Label', 'Feature': 'Feature'}
        plotter = self.SimplePlot(columns)
        testData = pd.DataFrame([['A', 1],
                                 ['A', 1],
                                 ['A', 2],
                                 ['A', 2],
                                 ['A', 2],
                                 ['A', 2],
                                 ['C', 0],
                                 ['C', 0],
                                 ['C', 0],
                                 ['C', 1],
                                 ['B', 3],
                                 ['B', 3],
                                 ['A', 'None'],
                                 ['A', 'None'],
                                 ['A', 'None'],
                                 ['A', 'None'],
                                 ['A', 'None'],
                                 ['B', 'None'],
                                 ],
                                columns=['Label', 'Feature'])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature')
        self.assertEqual(categories, [0, 3, 'None', 2, 1])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature', max_count=3)
        self.assertEqual(categories, [0, 3, 'None'])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature', max_count=2)
        self.assertEqual(categories, [0, 'None'])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature', max_count=2,
                                              includeNone=False)
        self.assertEqual(categories, [0, 3])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature', max_count=4,
                                              includeNone=False)
        self.assertEqual(categories, [0, 3, 2, 1])

        categories = plotter.selectCategories(testData, labelColumn='Label', featureColumn='Feature', max_count=40,
                                              includeNone=False)
        self.assertEqual(categories, [0, 3, 2, 1])


class Test_MultiBarDf(unittest.TestCase):
    class SimpleMultiBar(AbstractComplexBar):
        def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            return dataToProcess

        def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
            return dataToProcess

    def verifyResult(self, x: pd.DataFrame, y: pd.DataFrame):
        self.assertEqual(list(x.columns), list(y.columns))

        self.assertEqual(x.shape, y.shape)

        x = x.values.tolist()
        y = y.values.tolist()

        for m, row in enumerate(x):
            for n, col in enumerate(row):
                x_value = col
                y_value = y[m][n]

                if isinstance(x_value, numbers.Number):
                    self.assertTrue(isinstance(y_value, numbers.Number))
                    self.assertAlmostEqual(float(x_value), float(y_value), delta=0.001)

                else:
                    self.assertEqual(x_value, y_value)

    def setUp(self):
        self.columns = {'Label': 'Label', 'Feature': 'Feature'}
        self.solution_columns = ['Label', 'Feature', 'Size']
        self.plotter = self.SimpleMultiBar(self.columns)

        self.testData = pd.DataFrame([['A', '1'],
                                      ['A', '1'],
                                      ['A', '1'],
                                      ['A', '2'],

                                      ['C', '1'],
                                      ['C', '0'],

                                      ['B', '2'],
                                      ['B', '2'],
                                      ['B', 'None'],
                                      ],
                                     columns=['Label', 'Feature'])

    def test_empty(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=[], includeOther=False)
        self.verifyResult(df, pd.DataFrame(columns=['Label', 'Feature', 'Size']))

    def test_simple(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature')

        solution_data = [['A', '0', 0],
                         ['A', '1', 0.75],
                         ['A', '2', 0.25],
                         ['A', 'None', 0],

                         ['B', '0', 0],
                         ['B', '1', 0],
                         ['B', '2', 2 / 3],
                         ['B', 'None', 1 / 3],

                         ['C', '0', 0.5],
                         ['C', '1', 0.5],
                         ['C', '2', 0],
                         ['C', 'None', 0]
                         ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_with_categories(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['1', '2'])

        solution_data = [
            ['A', '1', 0.75],
            ['A', '2', 0.25],
            ['A', 'Other', 0],

            ['B', '1', 0],
            ['B', '2', 2 / 3],
            ['B', 'Other', 1 / 3],

            ['C', '1', 0.5],
            ['C', '2', 0],
            ['C', 'Other', 0.5]
        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)
        self.verifyResult(df, solution)

    def test_with_categories_no_other(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['1', '2'], includeOther=False)

        solution_data = [
            ['A', '1', 0.75],
            ['A', '2', 0.25],

            ['B', '1', 0],
            ['B', '2', 2 / 3],

            ['C', '1', 0.5],
            ['C', '2', 0],
        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)
        self.verifyResult(df, solution)

    def test_categories_ignoreNone(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', ignoreNone=True)

        solution_data = [['A', '0', 0],
                         ['A', '1', 0.75],
                         ['A', '2', 0.25],

                         ['B', '0', 0],
                         ['B', '1', 0],
                         ['B', '2', 2 / 3],

                         ['C', '0', 0.5],
                         ['C', '1', 0.5],
                         ['C', '2', 0],
                         ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_excludeOther_excludeNone(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['1', '2'], includeOther=False,
                                          ignoreNone=True)

        solution_data = [
            ['A', '1', 0.75],
            ['A', '2', 0.25],

            ['B', '1', 0],
            ['B', '2', 2 / 3],

            ['C', '1', 0.5],
            ['C', '2', 0],
        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_thresholding(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', threshold=0.6)

        solution_data = [
            ['A', '1', 0.75],
            ['A', '2', 0.0],

            ['B', '1', 0],
            ['B', '2', 2 / 3],

            ['C', '1', 0.0],
            ['C', '2', 0],

        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_thresholding_includeOther(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['0', '2'], includeOther=True,
                                          threshold=0.4)

        solution_data = [['A', '0', 0],
                         ['A', '2', 0],
                         ['A', 'Other', 0.75],

                         ['B', '0', 0],
                         ['B', '2', 2 / 3],
                         ['B', 'Other', 0],

                         ['C', '0', 0.5],
                         ['C', '2', 0],
                         ['C', 'Other', 0.5]
                         ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)
        self.verifyResult(df, solution)

    def test_with_None_as_category(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['None', '1'])

        solution_data = [
            ['A', '1', 0.75],
            ['A', 'None', 0],
            ['A', 'Other', 0.25],

            ['B', '1', 0],
            ['B', 'None', 1 / 3],
            ['B', 'Other', 2 / 3],

            ['C', '1', 0.5],
            ['C', 'None', 0],
            ['C', 'Other', 0.5],
        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_ignoreNone_with_None_as_category(self):
        df = self.plotter.createBarPlotDf(self.testData, 'Feature', categories=['None', '1'], ignoreNone=True)

        solution_data = [
            ['A', '1', 0.75],
            ['A', 'Other', 0.25],

            ['B', '1', 0],
            ['B', 'Other', 2 / 3],

            ['C', '1', 0.5],
            ['C', 'Other', 0.5],

        ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)

    def test_with_custom_sizes(self):
        sizes = self.testData.groupby("Label").size() * 0.5

        df = self.plotter.createBarPlotDf(self.testData, 'Feature', sizes=sizes)

        solution_data = [['A', '0', 0],
                         ['A', '1', 0.75 * 2],
                         ['A', '2', 0.25 * 2],
                         ['A', 'None', 0],

                         ['B', '0', 0],
                         ['B', '1', 0],
                         ['B', '2', 2 / 3 * 2],
                         ['B', 'None', 1 / 3 * 2],

                         ['C', '0', 0.5 * 2],
                         ['C', '1', 0.5 * 2],
                         ['C', '2', 0],
                         ['C', 'None', 0]
                         ]

        solution = pd.DataFrame(solution_data, columns=self.solution_columns)

        self.verifyResult(df, solution)


class test_Plots(unittest.TestCase):
    def verifyResult(self, x: pd.DataFrame, y: pd.DataFrame):
        self.assertEqual(list(x.columns), list(y.columns))

        self.assertEqual(x.shape, y.shape)

        x = x.values.tolist()
        y = y.values.tolist()

        for m, row in enumerate(x):
            for n, col in enumerate(row):
                x_value = col
                y_value = y[m][n]

                if isinstance(x_value, numbers.Number):
                    self.assertTrue(isinstance(y_value, numbers.Number))
                    self.assertAlmostEqual(float(x_value), float(y_value), delta=0.001)

                else:
                    self.assertEqual(x_value, y_value)

    def setUp(self):
        self.columns = {'Label': 'My_Label', 'Feature': 'My_Feature'}

    def test_createDataFrame(self):
        testData = [['A', '0'],
                    ['A', '01'],
                    ['A', '02'],
                    ['A', '07'],
                    ['A', 'None'],
                    ['B', '0'],
                    ['B', '01'],
                    ['A', '03'],
                    ['A', '08'],
                    ['A', '09'],
                    ['A', '10'],
                    ['A', '04'],
                    ['A', '05'],
                    ['A', '06'],
                    ['B', '01']
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Feature'])

        plotter = StackedBarPlot(self.columns)
        df = plotter.createDataFrame(testData)

        solutionData = [['A', '0', 1 / 12],
                        ['A', '01', 1 / 12],
                        ['A', '02', 1 / 12],
                        ['A', '03', 1 / 12],
                        ['A', '04', 1 / 12],
                        ['A', '05', 1 / 12],
                        ['A', '06', 1 / 12],
                        ['A', '07', 1 / 12],
                        ['A', 'None', 1 / 12],
                        ['A', 'Other', 0.25],
                        ['B', '0', 1 / 3],
                        ['B', '01', 2 / 3],
                        ['B', '02', 0.0],
                        ['B', '03', 0.0],
                        ['B', '04', 0.0],
                        ['B', '05', 0.0],
                        ['B', '06', 0.0],
                        ['B', '07', 0.0],
                        ['B', 'None', 0.0],
                        ['B', 'Other', 0.0]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Feature', 'Size'])

        self.verifyResult(df, solution)

    def test_mean_values_plot(self):
        testData = [['B', 0],
                    ['A', 0],
                    ['C', 1],
                    ['C', 3],
                    ['A', 1],
                    ['A', 2],
                    ['B', -2],
                    ]
        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Feature'])

        solutionData = [['A', 1.5],
                        ['B', 0],
                        ['C', 2]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Feature'])

        df = MeanValuesPlot(self.columns).createDataFrame(testData)
        self.verifyResult(df, solution)

    def test_number_of_packets(self):
        testData = [['B', 0],
                    ['A', 0],
                    ['C', 1],
                    ['C', 3],
                    ['A', 1],
                    ['A', 2],
                    ['B', -2],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Feature'])

        solutionData = [['A', 3],
                        ['B', 2],
                        ['C', 2]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'Count'])

        df = PlotNumberOfPackets(self.columns).createDataFrame(testData)
        self.verifyResult(df, solution)

    def test_address_interval_boxplot(self):
        testData = [['A', 'S1', 0],
                    ['A', 'S1', 200],
                    ['A', 'S2', 200],
                    ['B', 'S11', 0],
                    ['B', 'S11', 800],
                    ['B', 'S21', 900],
                    ['A', 'S2', 400],
                    ['A', 'S2', 700],
                    ['A', 'S3', 700],
                    ['A', 'S3', 1200],
                    ['B', 'S21', 1000],
                    ['B', 'S31', 1000],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Source', 'My_Time'])
        testData['My_Time'] = pd.to_datetime(testData['My_Time'], unit='s')

        solutionData = [['A', 'S1', 3 + 1 / 3],
                        ['A', 'S2', 8 + 1 / 3],
                        ['A', 'S3', 8 + 1 / 3],
                        ['B', 'S11', 13 + 1 / 3],
                        ['B', 'S21', 1 + 2 / 3],
                        ['B', 'S31', 0]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Source', 'My_Time'])

        df = (PlotAddressIntervalBoxPlot({'Label': 'My_Label', 'Source': 'My_Source', 'Time': 'My_Time'})
              .createDataFrame(testData))

        self.verifyResult(df, solution)

    def test_address_interval_mean(self):
        testData = [['A', 'S1', 0],
                    ['A', 'S1', 200],
                    ['A', 'S2', 200],
                    ['B', 'S11', 0],
                    ['B', 'S11', 800],
                    ['B', 'S21', 900],
                    ['A', 'S2', 400],
                    ['A', 'S2', 700],
                    ['A', 'S3', 700],
                    ['A', 'S3', 1200],
                    ['B', 'S21', 1000],
                    ['B', 'S31', 1000],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Source', 'My_Time'])
        testData['My_Time'] = pd.to_datetime(testData['My_Time'], unit='s')

        df = (PlotAddressIntervalMean({'Label': 'My_Label', 'Source': 'My_Source', 'Time': 'My_Time'})
              .createDataFrame(testData))

        solutionData = [['A', 6 + 2 / 3],
                        ['B', 5.0]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Time'])

        self.verifyResult(df, solution)

    def test_Broadcast(self):
        testData = [['A', 0],
                    ['A', 0],
                    ['B', 0],
                    ['A', 1],
                    ['A', 1],
                    ['B', 1],
                    ['B', 1],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Broadcast'])

        df = PlotBroadcast(columns={'Label': 'My_Label', 'Broadcast': 'My_Broadcast'}).createDataFrame(testData)

        solutionData = [['A', 0.5],
                        ['B', 2 / 3]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'Size'])

        self.verifyResult(df, solution)

    def test_Protocol(self):
        testData = [['A', 'P'],
                    ['A', 'L'],
                    ['B', 'P'],
                    ['A', 'L'],
                    ['A', 'None'],
                    ['B', 'None'],
                    ['B', 'None'],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Protocol'])

        df = PlotProtocol(columns={'Label': 'My_Label', 'Protocol': 'My_Protocol'}).createDataFrame(testData)

        solutionData = [['A', 'L', 0.5],
                        ['A', 'None', 0.25],
                        ['A', 'P', 0.25],
                        ['B', 'L', 0],
                        ['B', 'None', 2 / 3],
                        ['B', 'P', 1 / 3]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Protocol', 'Size'])

        self.verifyResult(df, solution)

    def test_advertisement(self):
        testData = [['1', 'A,B'],
                    ['1', 'None'],
                    ['2', 'B'],
                    ['2', 'None,C'],
                    ['1', 'A,C,A'],
                    ['2', 'C,C']]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_ADV'])

        df = PlotAdvertisementType(columns={'Label': 'My_Label', 'AD Type': 'My_ADV'}).createDataFrame(testData)

        solutionData = [['1', 'A', 1],
                        ['1', 'B', 1 / 3],
                        ['1', 'C', 1 / 3],
                        ['2', 'A', 0],
                        ['2', 'B', 1 / 3],
                        ['2', 'C', 1]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_ADV', 'Size'])

        self.verifyResult(df, solution)

    def test_UUID(self):
        testData = [['1', 'A,B'],
                    ['1', 'None'],
                    ['2', 'B'],
                    ['2', 'None,C'],
                    ['1', 'A,C,A,A'],
                    ['2', 'C,C']]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_ID'])

        df = PlotUUID(columns={'Label': 'My_Label', 'UUID': 'My_ID'}).createDataFrame(testData)

        solutionData = [['1', 'A', 4 / 3],
                        ['1', 'B', 1 / 3],
                        ['1', 'C', 1 / 3],
                        ['2', 'A', 0],
                        ['2', 'B', 1 / 3],
                        ['2', 'C', 1]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_ID', 'Size'])

        self.verifyResult(df, solution)

    def test_MalformedPacket(self):
        testData = [['1', 1],
                    ['1', 0],
                    ['2', 1],
                    ['2', 1],
                    ['1', 3],
                    ['2', 1]]

        testData = pd.DataFrame(testData, columns=['My_Label', 'My_Mal'])

        df = PlotMalformedPacket(columns={'Label': 'My_Label', 'Malformed': 'My_Mal'}).createDataFrame(testData)

        solutionData = [['1', False, 1 / 3],
                        ['1', True, 2 / 3],
                        ['2', False, 0],
                        ['2', True, 1]]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Mal', 'Size'])

        self.verifyResult(df, solution)

    def test_packet_rate_mean(self):
        # Time, Label, Channel, File
        testData = [[10, 'A', 37, '1'],
                    [30, 'A', 37, '1'],
                    [60, 'A', 37, '1'],
                    [90, 'A', 37, '1'],
                    [120, 'A', 37, '1'],
                    [150, 'A', 37, '1'],
                    [180, 'A', 37, '1'],
                    [1, 'A', 38, '1'],
                    [30, 'A', 38, '1'],
                    [30, 'A', 39, '2'],
                    [40, 'A', 39, '2'],
                    [70, 'B', 37, '1'],
                    [80, 'B', 37, '1'],
                    [180, 'B', 37, '1'],
                    [130, 'B', 37, '2'],
                    [40, 'A', 38, '1'],
                    [50, 'A', 38, '1'],
                    [70, 'A', 38, '1'],
                    [100, 'A', 38, '1'],
                    [110, 'A', 38, '1'],
                    [20, 'A', 39, '2'],
                    [90, 'A', 39, '2'],

                    [100, 'B', 0, '2'],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Time', 'My_Label', 'My_Channel', 'My_File'])
        testData['My_Time'] = pd.to_datetime(testData['My_Time'], unit='s')

        df = (PlotPacketRateMean({'Time': 'My_Time', 'Label': 'My_Label', 'Channel': 'My_Channel', 'File': 'My_File'})
              .createDataFrame(testData))

        solutionData = [['A', 'CH 37', (2 / 60 + 2 / 60 + 2 / 60 + 1 / 60) / 4],
                        ['A', 'CH 38', (4 / 60 + 3 / 60) / 2],
                        ['A', 'CH 39', (3 / 60 + 1 / 60) / 2],

                        ['B', 'CH 37', ((2 / 60 + 0 / 60 + 1 / 60) / 3 + 1 / 60) / 2],
                        ['B', 'CH 38', 0],
                        ['B', 'CH 39', 0],
                        ]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'My_Channel', 'Rate'])
        self.verifyResult(df, solution)

    def test_packet_rate_boxplot(self):
        # Time, Label, Channel, File
        testData = [[5, 'A', 37, '1'],
                    [30, 'A', 37, '1'],
                    [70, 'A', 37, '1'],

                    [70, 'A', 38, '1'],
                    [80, 'A', 38, '1'],
                    [190, 'A', 38, '1'],
                    [90, 'A', 38, '2'],

                    [90, 'B', 39, '3'],
                    [100, 'B', 1039, '3'],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Time', 'My_Label', 'My_Channel', 'My_File'])
        testData['My_Time'] = pd.to_datetime(testData['My_Time'], unit='s')

        df = (PlotPacketRateBoxPlot({'Time': 'My_Time', 'Label': 'My_Label', 'Channel': 'My_Channel', 'File': 'My_File'})
            .createDataFrame(testData))

        solutionData = [['A CH 37', (2 / 60)],
                        ['A CH 37', (1 / 60)],
                        ['A CH 38', (2 / 60)],
                        ['A CH 38', 0],
                        ['A CH 38', (1 / 60)],
                        ['A CH 38', (1 / 60)],
                        ['B CH 39', (1 / 60)],
                        ]

        solution = pd.DataFrame(solutionData, columns=['My_Label', 'Rate'])
        self.verifyResult(df, solution)

    def test_packet_rate_graph(self):
        # Time, Label, Channel, File
        testData = [[0, 'A', 37, '1'],
                    [30, 'A', 37, '1'],
                    [80, 'A', 38, '1'],
                    [190, 'A', 38, '1'],
                    [70, 'A', 37, '1'],
                    [90, 'B', 39, '3'],
                    [70, 'A', 38, '1'],
                    [90, 'A', 38, '2'],
                    [100, 'B', 1039, '3'],
                    ]

        testData = pd.DataFrame(testData, columns=['My_Time', 'My_Label', 'My_Channel', 'My_File'])
        testData['My_Time'] = pd.to_datetime(testData['My_Time'], unit='s')

        df = (PlotPacketRateGraph({'Time': 'My_Time', 'Label': 'My_Label', 'Channel': 'My_Channel', 'File': 'My_File'})
              .createDataFrame(testData))

        solutionData = [['1', 'A', 37, 0, (2 / 60)],
                        ['1', 'A', 37, 1, (1 / 60)],
                        ['1', 'A', 38, 1, (2 / 60)],
                        ['1', 'A', 38, 2, 0],
                        ['1', 'A', 38, 3, (1 / 60)],
                        ['2', 'A', 38, 0, (1 / 60)],
                        ['3', 'B', 39, 0, (1 / 60)],
                        ]

        solution = pd.DataFrame(solutionData, columns=['My_File', 'My_Label', 'My_Channel', 'My_Time', 'Rate'])
        solution['My_Time'] = pd.to_timedelta(solution['My_Time'], unit='s')
        solution['My_Time'] = solution['My_Time'].dt.total_seconds() / 60

        self.verifyResult(df, solution)
