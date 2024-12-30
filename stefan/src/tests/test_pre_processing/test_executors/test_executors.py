import random
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler

from src.pre_processing.executors.executors import *
from src.pre_processing.tasks.task import Task


class Test_executors(unittest.TestCase):
    def verifyResult(self, x: pd.DataFrame, y: pd.DataFrame):
        self.assertEqual(list(x.columns), list(y.columns))

        for column in x.columns:
            self.assertEqual(x[column].to_numpy().tolist(), y[column].to_numpy().tolist())

    def test_toDatetime(self):
        current_time = 1708617550
        targetTime = '2024-02-22 15:59:10'

        testData = pd.DataFrame([[current_time, 3]], columns=['A', 'B'])

        executor = toDateTime(column='A')
        task = Task('TestTask', priority=1, executor=executor)

        testData = task.execute(testData)

        convertedTime = str(testData.iloc[0, 0])
        self.assertEqual(convertedTime, targetTime)

    def test_renameColumns(self):
        testData = pd.DataFrame([[1, 3]], columns=['A', 'B'])
        executor = renameColumns(columns={'A': 'C'})
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual(['C', 'B'], list(testData.columns))

    def test_dropColumns(self):
        testData = pd.DataFrame([[1, 3]], columns=['A', 'B'])
        executor = dropColumns(columns=['A'])
        task = Task('TestTask', priority=1, executor=executor)

        testData = task.execute(testData)

        self.assertEqual(['B'], list(testData.columns))

    def test_convertDataType(self):
        testData = pd.DataFrame([[1.4, 3, 'Test', 4]], columns=['A', 'B', 'C', 'D'])
        executor = convertDataType({'A': 'float32', 'B': 'int16', 'C': 'string', 'D': 'string'})
        task = Task('TestTask', priority=1, executor=executor)

        testData = task.execute(testData)

        self.assertEqual(testData.A.dtype, 'float32')
        self.assertEqual(testData.B.dtype, 'int16')
        self.assertEqual(testData.C.dtype, 'string')
        self.assertEqual(testData.D.dtype, 'string')

    def test_keepColumns(self):
        testData = pd.DataFrame([[1.4, 3, 'Test', 4]], columns=['A', 'B', 'C', 'D'])

        executor = keepColumns(columns=['C', 'B'])
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)
        self.assertEqual(['C', 'B'], list(testData.columns))

    def test_fillNa(self):
        testData = pd.DataFrame([[3, np.nan, None, 'Test', None]], columns=['A', 'B', 'C', 'D', 'E'])
        executor = fillNa(fillValue='Value', columns=['C', 'B', 'D'])
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual([[3, 'Value', 'Value', 'Test', None]], testData.to_numpy().tolist())

        testData = pd.DataFrame([[3, np.nan, None, 'Test', np.nan]], columns=['A', 'B', 'C', 'D', 'E'])
        executor = fillNa(fillValue='Value')
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual([[3, 'Value', 'Value', 'Test', 'Value']], testData.to_numpy().tolist())

    def test_customColumnFunction(self):
        testData = pd.DataFrame([[1, 3]], columns=['A', 'B'])

        executor = customColumnFunction(column='B', function=lambda x: x * 2 + 1)
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual([[1, 7]], testData.to_numpy().tolist())

    def test_convertObjectToString(self):
        testData = pd.DataFrame([['Test', 3], [1, 2]], columns=['A', 'B'])

        executor = convertObjectToString()
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual(testData.A.dtype, 'string')

    def test_order_columns(self):
        testData = pd.DataFrame([[1, 3], [-1, 2], [2, 4]], columns=['A', 'B'])
        executor = orderDataFrame(column='A', ascending=False)

        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.assertEqual([[2, 4], [1, 3], [-1, 2]], testData.to_numpy().tolist())

    def test_labeling(self):
        testData = pd.DataFrame([[1, 3], [3, 4]], columns=['A', 'B'])
        executor = labeling(label='TEST', labelColumn='MY_LABEL')

        solution = testData.copy()
        solution['MY_LABEL'] = 'TEST'

        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.verifyResult(solution, testData)

    def test_autoLabeling(self):
        testData = pd.DataFrame(np.random.uniform(100, 101, size=(3, 4)), columns=['A', 'B', 'C', 'D'])

        trainData = testData.copy()
        trainData = trainData[['A', 'C']]
        trainLabel = pd.DataFrame(['AA', 'BB', 'AA'], columns=['LABEL'])

        solution = testData.copy()
        solution['LABEL'] = 'AA'

        dummy_clf = DummyClassifier(strategy="most_frequent")
        scaler = MinMaxScaler().fit(np.ones((3, 2)))
        dummy_clf.fit(trainData, trainLabel)

        executor = autoLabel(dummy_clf, scaler, dropColumns=['B', 'D'], labelColumn='LABEL')
        task = Task('TestTask', priority=1, executor=executor)
        testData = task.execute(testData)

        self.verifyResult(solution, testData)

    def test_model_rate(self):
        random.seed(0) # do not change!

        time = [0, 1, 3, 5, 6, 7, 8, 9, 10, 7, 21, 22, 23]
        source = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'C', 'C']
        label = [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 'L1', 'L2']
        feature = [2, 0, 0, 3, 2, 3, 0, 3, 3, 1, 1, 2, 4]

        data = pd.DataFrame({'Time': time, 'Source': source, 'Feature': feature, 'Label': label})
        data['Time'] = pd.to_datetime(data['Time'], unit='s')

        result = modelRate(time='Time', source='Source', seconds=5).execute(data)

        time_solution = [0, 5, 10, 15, 20, 5, 20]
        source_solution = ['A', 'A', 'A', 'A', 'A', 'B', 'C']
        label_solution = [1, 1, 1, 1, 1, 2, 'L2']
        feature_solution = [0.4, 1.6, 0.6, 0.0, 0.2, 0.8, 1.2]

        solution = pd.DataFrame(
            {'Time': time_solution, 'Source': source_solution, 'Feature': feature_solution, 'Label': label_solution})
        solution['Time'] = pd.to_datetime(solution['Time'], unit='s')

        print(result)

        self.verifyResult(result, solution)

    def test_malformed_packet(self):
        col = ['Test', 'testMalformedtest', 'malformed', 'test']
        data = pd.DataFrame({'TEST': col})

        result = malformedPacket(column='TEST', newColumn='NEW', malformed='Malformed').execute(data)

        col = ['Test', 'testMalformedtest', 'malformed', 'test']
        solution = pd.DataFrame({'TEST': col, 'NEW': [0, 1, 0, 0]})

        self.verifyResult(result, solution)

    def test_multipleDevices_simple(self):
        source = ['A', 'AA', 'A', 'B', 'AA', 'B', 'A', 'BB', 'A', 'C', 'B']

        data = pd.DataFrame({'Source': source})

        result = labelMultipleDevices(firstDevice=1, secondDevice=2, labelColumn='MyLabel').execute(data)

        source_solution = ['AA', 'A', 'B', 'AA', 'B', 'A', 'BB', 'A']
        label_solution = [2, 1, 2, 2, 2, 1, 2, 1]

        solution = pd.DataFrame({'Source': source_solution, 'MyLabel': label_solution})

        self.verifyResult(result, solution)

    def test_multipleDevices_slicing(self):
        source = ['AA', 'A', 'AA', 'A', 'B', 'AA', 'B', 'A', 'BB', 'A', 'C', 'B']

        data = pd.DataFrame({'Source': source})

        result = labelMultipleDevices(firstDevice=1, secondDevice=2, labelColumn='MyLabel',
                                      selection=slice(0, 2)).execute(data)

        source_solution = ['B', 'AA', 'B', 'A', 'BB', 'A']
        label_solution = [2, 1, 2, 1, 2, 1]

        solution = pd.DataFrame({'Source': source_solution, 'MyLabel': label_solution})

        self.verifyResult(result, solution)

    def test_extractAppleDataType(self):
        company_col = ['Apple', '', 'Apple', 'Apple', 'Samsung', 'Samsung']
        data_col = ['1200', '1200', '1600', '', '1500', '1200']

        testData = pd.DataFrame({'Company': company_col, 'Data': data_col})

        result = extractAppleDataType(dataColumn='Data',
                                      company='Apple',
                                      companyColumn='Company',
                                      typeColumn='Type',
                                      noneValue='',
                                      ).execute(testData)

        type_col = ['12', '', '16', '', '', '']

        solution = pd.DataFrame({'Company': company_col, 'Data': data_col, 'Type': type_col})

        self.verifyResult(result, solution)

    def test_convertToBitLength(self):
        testData = pd.DataFrame({'Data': ['321', '21', np.nan, '']})

        solution = pd.DataFrame({'Data': ['321', '21', np.nan, ''], 'Length': [12, 8, 0, 0]})

        result = convertToBitLength(column='Data', newColumn='Length', noneValue='').execute(testData)

        self.verifyResult(result, solution)

    def test_addStateContinuityType(self):
        label_col = ['iPhone', 'Samsung', 'iPad', 'iPhone', 'iPhone', 'iPhone']

        continuity_col = ['12', '12', '12', '14', '16', 'None']
        length_col = ['4', '4', '8', '7', '3', '8']

        testData = pd.DataFrame({'Label': label_col, 'Continuity': continuity_col, 'Length': length_col})
        testData['State'] = ''

        solution = testData.copy(deep=True)
        solution['State'] = ['FindMy online', '', 'FindMy offline', 'CT 14', '', '']

        result = addStateContinuityType(labelColumn="Label",
                                        typeColumn="Continuity",
                                        types=['12', '14'],
                                        MS_DataColumn_Length='Length',
                                        strike=6,
                                        seperator='   ',
                                        stateColumn='State',
                                        labels=['iPhone', 'iPad']).execute(testData)

        self.verifyResult(result, solution)

    def test_instantiateStateColumn(self):
        testData = pd.DataFrame({'Col': [1, 2, 3]})
        solution = testData.copy(deep=True)
        solution['State'] = ''

        result = instantiateStateColumn(stateColumn='State').execute(testData)
        self.verifyResult(result, solution)

    def test_addState(self):
        testData = pd.DataFrame({'Col': [1, 2, 3], 'MyLabel': ['A', 'B', 'C']})
        testData['State'] = ''

        solution = testData.copy(deep=True)
        solution['State'] = ['Z', '', 'Z']

        result = addState(stateColumn='State', stateLabel='Z', labels=['A', 'C'], labelColumn='MyLabel').execute(
            testData)
        self.verifyResult(result, solution)

    def test_addState_strip(self):
        testData = pd.DataFrame({'Col': [1, 2, 3]})
        testData['State'] = ['B', 'B', 'B']

        solution = testData.copy(deep=True)
        solution['State'] = ['B  A', 'B  A', 'B  A']

        result = addState(stateColumn='State', stateLabel='A', seperator='  ').execute(testData)
        self.verifyResult(result, solution)

    def test_collapseStateColumn(self):
        testData = pd.DataFrame({'Col': [1, 2], 'Label': ['A', 'A'], 'State': ['B', 'B']})

        solution = pd.DataFrame({'Col': [1, 2], 'Label': ['AZB', 'AZB']})

        result = collapseStateColumn(labelColumn='Label', stateColumn='State', seperator='Z').execute(testData)
        self.verifyResult(result, solution)

    def test_StringReplace(self):
        testData = pd.DataFrame({'A': [1, 'AC', 3, 4], 'B': ['AB,AC,CC', 'CC', 'AB', '']})

        solution = pd.DataFrame({'A': [1, 'AC', 3, 4], 'B': ['XX,YY,CC', 'CC', 'XX', '']})

        result = StringReplace(column='B', replace={'AB': 'XX', 'AC': 'YY'}).execute(testData)
        self.verifyResult(result, solution)

    def test_dropLabels(self):
        testData = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['A', 'B', 'ABA', ' ']})

        solution = pd.DataFrame({'A': [2, 3, 4], 'B': ['B', 'ABA', ' ']})

        result = dropLabels(labelColumn='B', labels=['A']).execute(testData)
        self.verifyResult(result, solution)

    def test_extractSamsungDataType(self):
        company_col = ['Samsung', 'None', 'Apple', 'Samsung', 'Samsung', 'Samsung', 'Samsung']
        data_col = ['1F00', '1200', '1500', '', '1313000', 'None', '4900']

        testData = pd.DataFrame({'Company': company_col, 'Data': data_col})

        result = extractSamsungDataType(dataColumn='Data',
                                        company='Samsung',
                                        companyColumn='Company',
                                        typeColumn='Type',
                                        noneValue='None',
                                        ).execute(testData)

        type_col = ['7', 'None', 'None', 'None', '3', 'None', '1']

        solution = pd.DataFrame({'Company': company_col, 'Data': data_col, 'Type': type_col})
        self.verifyResult(result, solution)

    def test_cleanPDU(self):
        testData = pd.DataFrame(['ADV_IND[Malformed Packet]',
                                 'ADV_NONCONN_IND[Malformed Packet]',
                                 'ADV_IND'],
                                columns=['PDU']
                                )
        solution = pd.DataFrame(['ADV_IND',
                                 'ADV_NONCONN_IND',
                                 'ADV_IND'],
                                columns=['PDU']
                                )
        result = cleanPDU('PDU').execute(testData)
        self.verifyResult(result, solution)


class Test_createDummies(unittest.TestCase):
    def setUp(self):
        self.prefix = 'T'
        self.column = 'TEST'
        self.simpleLabels = ['A', 'B', 'C']
        self.complexLabels = {'W': 'A', 'X': 'B', 'Y': 'C', 'Z': 'D'}
        self.prefix_sep = ' '

    def createSimpleDataFrame(self):
        dataList = ['B', 'A', '', 'C', 'B']
        return pd.DataFrame(dataList, columns=[self.column])

    def createComplexDataFrame(self):
        dataList = [['X,Y'], [''], ['W'], ['Y,Z,Z'], ['Y,Y']]
        return pd.DataFrame(dataList, columns=[self.column])

    def verifyResult(self, resultData, targetData, targetLabels, OtherColumn=False, NoneColumn=False,
                     OtherValue='Other', NoneValue='', seperator=' '):

        def get_column(value, customPrefix=self.prefix, customSep=seperator):
            return customPrefix + customSep + str(value)

        targetColumns = [get_column(x) for x in targetLabels]

        if OtherColumn:
            targetColumns.append(get_column(OtherValue))

        if NoneColumn:
            targetColumns.append(get_column(NoneValue))

        self.assertEqual(resultData.columns[0], self.column)
        self.assertEqual(list(resultData.columns[1:]), targetColumns)

        self.assertEqual(len(targetData), len(targetColumns))
        self.assertEqual(len(targetData) + 1, resultData.shape[1])

        for index, col in enumerate(targetData):
            self.assertEqual(resultData.iloc[:, index + 1].to_list(), col)

    def test_simpleDataFrame(self):
        testData = self.createSimpleDataFrame()
        task = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False)
        task = Task('TestTask', priority=1, executor=task)
        testData = task.execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels)
        self.assertEqual(1, task.getPriority())

    def test_dropColumn(self):
        testData = self.createSimpleDataFrame()
        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 dropColumn=True)

        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        self.assertEqual(['T A', 'T B', 'T C'], list(testData.columns))

    def test_simpleDataFrame_with_NoneColumn(self):
        testData = self.createSimpleDataFrame()
        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 NoneColumn=True)

        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels, NoneColumn=True)

    def test_simpleDataFrame_with_NoneColumn_and_Custom_NoneValue(self):
        myNone = 'myNone'
        testData = self.createSimpleDataFrame()
        testData.iloc[2, 0] = myNone

        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 NoneColumn=True, NoneValue='myNone')
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels, NoneColumn=True, NoneValue='myNone')

    def test_simpleDataFrame_without_NoneColumn_and_Custom_NoneValue(self):
        myNone = 'myNone'
        testData = self.createSimpleDataFrame()
        testData.iloc[2, 0] = myNone

        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 NoneColumn=False, NoneValue=myNone)

        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels, NoneColumn=False, NoneValue=myNone)

    def test_simpleDataFrame_wit_OtherColumn_and_Custom_OtherValue(self):
        myOther = 'myOther'
        testData = self.createSimpleDataFrame()

        executor = createDummies(self.column, self.simpleLabels[1:], self.prefix, splitting=False,
                                 OtherColumn=True, OtherValue=myOther)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels[1:], OtherColumn=True, OtherValue=myOther)

    def test_simpleDataFrame_with_OtherColumn_but_missing_Value(self):
        testData = self.createSimpleDataFrame()

        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 OtherColumn=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels, OtherColumn=True)

    def test_simpleDataFrame_with_OtherColumn_and_NoneColumn(self):
        testData = self.createSimpleDataFrame()

        executor = createDummies(self.column, self.simpleLabels[1:], self.prefix, splitting=False,
                                 OtherColumn=True, NoneColumn=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels[1:], OtherColumn=True, NoneColumn=True)

    def test_simpleDataFrame_without_NoneColumn_and_with_OtherColum_and_CustomNone(self):
        myNone = 'myNone'
        testData = self.createSimpleDataFrame()
        testData.iloc[2, 0] = myNone

        executor = createDummies(self.column, self.simpleLabels[1:], self.prefix, splitting=False,
                                 OtherColumn=True, NoneValue=myNone)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[1, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels[1:], OtherColumn=True, NoneValue=myNone)

    def test_simpleDataFrame_with_dict(self):
        dataList = ['Y', 'X', 'None', 'Z', 'Y']
        testData = pd.DataFrame(dataList, columns=[self.column])

        executor = createDummies(self.column, self.complexLabels, self.prefix,
                                 splitting=False)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]

        self.verifyResult(testData, targetData, self.complexLabels.values())

    def test_simpleDataFrame_with_fewer_labels(self):
        testData = self.createSimpleDataFrame()
        testData.iloc[3, 0] = 'A'

        executor = createDummies(self.column, self.simpleLabels, self.prefix,
                                 splitting=False)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 1, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels)

    def test_emptyDataFrame(self):
        testData = pd.DataFrame([], columns=[self.column])
        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 OtherColumn=True, NoneColumn=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[], [], [], [], []]

        self.verifyResult(testData, targetData, self.simpleLabels, OtherColumn=True, NoneColumn=True)

    def test_with_Integers(self):
        dataList = ['B', 1, 'None', 3, 'B']
        testData = pd.DataFrame(dataList, columns=[self.column])
        labels = [1, 'B', 3]

        executor = createDummies(self.column, labels, self.prefix, splitting=False)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]

        self.verifyResult(testData, targetData, labels)

    def test_simpleDataFrame_with_custom_seperator(self):
        testData = self.createSimpleDataFrame()
        executor = createDummies(self.column, self.simpleLabels, self.prefix, splitting=False,
                                 prefix_sep=':')
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 1, 0]]

        self.verifyResult(testData, targetData, self.simpleLabels, seperator=':')

    def test_complexDataFrame_V1(self):
        testData = self.createComplexDataFrame()
        executor = createDummies(self.column, self.complexLabels, self.prefix,
                                 splitting=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 2], [0, 0, 0, 2, 0]]

        self.verifyResult(testData, targetData, self.complexLabels.values())

    def test_complexDataFrame_V2(self):
        testData = self.createComplexDataFrame()
        customLabels = {'W': 'A', 'X': 'B', 'Y': 'C'}
        testData.iloc[2, 0] = 'Z,'
        executor = createDummies(self.column, customLabels, self.prefix, splitting=True, OtherColumn=True,
                                 NoneColumn=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 2], [0, 0, 1, 2, 0], [0, 1, 1, 0, 0]]

        self.verifyResult(testData, targetData, customLabels.values(), OtherColumn=True, NoneColumn=True)

    def test_complexDataFrame_V3(self):
        testData = self.createComplexDataFrame()
        customLabels = {'W': 'A', 'X': 'B', 'Y': 'C'}
        testData.iloc[2, 0] = 'Z'
        executor = createDummies(self.column, customLabels, self.prefix, splitting=True, OtherColumn=False,
                                 NoneColumn=True)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 2], [0, 1, 0, 0, 0]]

        self.verifyResult(testData, targetData, customLabels.values(), OtherColumn=False, NoneColumn=True)

    def test_complexDataFrame_v4(self):
        testData = self.createComplexDataFrame()
        customLabels = {}
        executor = createDummies(self.column, customLabels, self.prefix, splitting=True, OtherColumn=False,
                                 NoneColumn=False)
        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = []

        self.verifyResult(testData, targetData, customLabels.values(), OtherColumn=False, NoneColumn=False)

    def test_complex_splitting(self):
        dataList = [['X,X,Y'], ['X,X'], ['Y,A,B,Z,A,B'], ['A,A,B,B']]
        customLabels = {'Y': 'Y', 'X,X': 'X', 'A,B': 'C'}
        testData = pd.DataFrame(dataList, columns=[self.column])

        executor = createDummies(self.column, customLabels, self.prefix, splitting=True, OtherColumn=True,
                                 NoneColumn=False)

        testData = Task('TestTask', priority=1, executor=executor).execute(testData)
        targetData = [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]]

        self.verifyResult(testData, targetData, customLabels.values(), OtherColumn=True, NoneColumn=False)
