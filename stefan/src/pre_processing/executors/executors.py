import random
from typing import Callable

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype

from src.pre_processing.interfaces.executor_interface import ExecutorInterface


class toDateTime(ExecutorInterface):
    def __init__(self, column: str, unit: str = 's'):
        self.column = column
        self.unit = unit

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        temp = pd.to_datetime(dataToProcess[self.column], unit=self.unit)
        dataToProcess[self.column] = dataToProcess.loc[:, self.column].astype(object)
        dataToProcess.loc[:, self.column] = temp
        dataToProcess[self.column] = dataToProcess.loc[:, self.column].astype("datetime64[ns]")

        return dataToProcess


class renameColumns(ExecutorInterface):
    def __init__(self, columns: dict[str:str]):
        self.columns = columns

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess = dataToProcess.rename(self.columns, axis=1)
        return dataToProcess


class dropColumns(ExecutorInterface):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess.drop(self.columns, inplace=True, axis=1)
        return dataToProcess


class convertDataType(ExecutorInterface):
    def __init__(self, columns: dict[str:str]):
        self.columns = columns

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess = dataToProcess.astype(self.columns)
        return dataToProcess


class convertObjectToString(ExecutorInterface):
    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        string_columns = []

        for column in dataToProcess.columns:
            if is_object_dtype(dataToProcess[column]):
                string_columns.append(column)

        string_columns_dict = {column: "string" for column in string_columns}

        dataToProcess = dataToProcess.astype(string_columns_dict)
        return dataToProcess


class keepColumns(ExecutorInterface):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess = dataToProcess[self.columns]
        return dataToProcess


class fillNa(ExecutorInterface):
    def __init__(self, fillValue, columns: list[str] = None):
        self.columns = columns
        self.fillValue = fillValue

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            dataToProcess = dataToProcess.fillna(self.fillValue)
        else:
            dataToProcess[self.columns] = dataToProcess[self.columns].fillna(self.fillValue)

        return dataToProcess


class autoLabel(ExecutorInterface):
    def __init__(self, classifier, scaler, dropColumns: list[str] = None, labelColumn: str = 'Label'):
        if dropColumns is None:
            dropColumns = []
        self.model = classifier
        self.scaler = scaler
        self.dropColumns = dropColumns
        self.label = labelColumn

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        X = dataToProcess.copy()

        if len(self.dropColumns) > 0:
            X.drop(self.dropColumns, inplace=True, axis=1)

        X = self.scaler.transform(X)
        dataToProcess[self.label] = self.model.predict(X)

        return dataToProcess


class labeling(ExecutorInterface):
    def __init__(self, label, labelColumn: str = 'Label'):
        self.column = labelColumn
        self.label = label

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.column] = self.label
        return dataToProcess


class labelMultipleDevices(ExecutorInterface):
    def __init__(self, firstDevice, secondDevice, selection: slice = slice(0), source: str = 'Source',
                 labelColumn: str = 'Label'):
        self.firstDevice = firstDevice
        self.secondDevice = secondDevice
        self.source = source
        self.label = labelColumn
        self.selection = selection

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        firstAddress = list(dataToProcess.loc[self.selection, [self.source]][self.source].unique())

        firstIndex = dataToProcess.index[~dataToProcess[self.source].isin(firstAddress)].min()
        lastIndex = dataToProcess.index[dataToProcess[self.source].isin(firstAddress)].max()

        dataToProcess = dataToProcess.iloc[firstIndex:lastIndex + 1, :]

        dataToProcess[self.label] = self.secondDevice
        dataToProcess.loc[dataToProcess[self.source].isin(firstAddress), self.label] = self.firstDevice

        return dataToProcess


class malformedPacket(ExecutorInterface):
    def __init__(self, column: str, newColumn: str, malformed: str = "Malformed Packet"):
        self.column = column
        self.newColumn = newColumn
        self.malformed = malformed

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.newColumn] = dataToProcess[self.column].str.contains(self.malformed).astype(int)
        return dataToProcess


class modelRate(ExecutorInterface):
    def __init__(self, time: str, source: str, seconds: int, labelColumn: str = 'Label'):
        self.time = time
        self.source = source
        self.seconds = seconds
        self.label = labelColumn

    def __getLabels(self, dataToProcess):
        labels = dataToProcess.groupby(self.source)[self.label].agg(pd.Series.mode).reset_index()

        def filterLabel(x):
            if isinstance(x, np.ndarray):
                return random.choice(x)
            else:
                return x

        labels[self.label] = labels[self.label].apply(filterLabel)

        return labels

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        labels = self.__getLabels(dataToProcess)

        temp = dataToProcess.copy(deep=True).drop([self.label], axis=1)
        columns = dataToProcess.columns
        temp = temp.set_index(self.time)

        temp = (temp.groupby(self.source).resample(str(self.seconds) + "s").sum().drop(self.source, axis=1) / self.seconds).reset_index()
        temp = temp.merge(labels, on=self.source, how='inner')
        temp = temp[columns]
        return temp


class extractAppleDataType(ExecutorInterface):
    def __init__(self, companyColumn: str, company: str, dataColumn: str, typeColumn: str, noneValue: str):
        self.companyColumn = companyColumn
        self.dataColumn = dataColumn
        self.typeColumn = typeColumn
        self.company = company
        self.noneValue = noneValue

    def __extractType(self, row):
        if self.company not in str(row[self.companyColumn]):
            return self.noneValue

        else:
            MS_Data = str(row[self.dataColumn])

            if MS_Data != '' and MS_Data != self.noneValue:
                return str(MS_Data[:2])
            else:
                return self.noneValue

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.typeColumn] = dataToProcess.apply(self.__extractType, axis=1)
        dataToProcess[self.typeColumn] = dataToProcess[self.typeColumn].astype("string")

        return dataToProcess


class convertToBitLength(ExecutorInterface):
    def __init__(self, column: str, newColumn: str, noneValue: str = ''):
        self.column = column
        self.newColumn = newColumn
        self.noneVale = noneValue

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess.loc[:, self.newColumn] = dataToProcess[self.column].fillna(self.noneVale).astype(str).apply(
            lambda x: len(x) * 4)
        dataToProcess[self.newColumn] = dataToProcess[self.newColumn].astype(int)
        return dataToProcess


class addStateContinuityType(ExecutorInterface):
    def __init__(self, labels: list, typeColumn: str, types: list, MS_DataColumn_Length: str, strike: int,
                 stateColumn: str = 'State', labelColumn: str = 'Label', seperator: str = ' '):
        self.typeColumn = typeColumn
        self.labels = labels
        self.labelColumn = labelColumn
        self.stateColumn = stateColumn
        self.MS_DataColumn_Length = MS_DataColumn_Length
        self.strike = strike
        self.seperator = seperator
        self.types = types

    def __label(self, row):
        currentLabel = str(row[self.labelColumn])
        currentType = str(row[self.typeColumn])
        currentState = str(row[self.stateColumn])

        if currentLabel not in self.labels or currentType not in self.types:
            return currentState

        if currentType == '12':
            if int(row[self.MS_DataColumn_Length]) < self.strike:
                return currentState + self.seperator + 'FindMy online'
            else:
                return currentState + self.seperator + 'FindMy offline'

        else:
            return currentState + self.seperator + 'CT' + ' ' + currentType

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.stateColumn] = dataToProcess.apply(self.__label, axis=1)
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].astype("string")
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].str.strip()
        return dataToProcess


class instantiateStateColumn(ExecutorInterface):
    def __init__(self, stateColumn: str):
        self.stateColumn = str(stateColumn)

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.stateColumn] = ""
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].astype("string")

        return dataToProcess


class collapseStateColumn(ExecutorInterface):
    def __init__(self, stateColumn: str, labelColumn: str = 'Label', seperator: str = ' '):
        self.stateColumn = str(stateColumn)
        self.labelColumn = str(labelColumn)
        self.seperator = str(seperator)

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].astype("string")
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].str.strip()

        dataToProcess[self.labelColumn] = dataToProcess[self.labelColumn] + self.seperator + dataToProcess[
            self.stateColumn]
        dataToProcess[self.labelColumn] = dataToProcess[self.labelColumn].astype("string")
        dataToProcess[self.labelColumn] = dataToProcess[self.labelColumn].str.strip()
        dataToProcess.drop(self.stateColumn, axis=1, inplace=True)

        return dataToProcess


class addState(ExecutorInterface):
    def __init__(self, stateLabel: str, stateColumn: str, labels: list[str] = None, labelColumn='Label', seperator=' '):
        if labels is None:
            self.labels = []
        else:
            self.labels = labels
        self.stateColumn = str(stateColumn)
        self.stateLabel = str(stateLabel)
        self.labelColumn = str(labelColumn)
        self.seperator = seperator

    def __label(self, row):
        if len(self.labels) == 0 or str(row[self.labelColumn]) in self.labels:
            return str(row[self.stateColumn]) + self.seperator + self.stateLabel

        else:
            return str(row[self.stateColumn])

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.stateColumn] = dataToProcess.apply(self.__label, axis=1)
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].astype("string")
        dataToProcess[self.stateColumn] = dataToProcess[self.stateColumn].str.strip()

        return dataToProcess


class StringReplace(ExecutorInterface):
    def __init__(self, column: str, replace: dict[str, str]):
        self.column = column
        self.replace = replace

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        for key in self.replace:
            dataToProcess[self.column] = dataToProcess[self.column].str.replace(key, self.replace[key])

        return dataToProcess


class extractSamsungDataType(ExecutorInterface):
    def __init__(self, companyColumn: str, company: str, dataColumn: str, typeColumn: str, noneValue: str):
        self.companyColumn = companyColumn
        self.dataColumn = dataColumn
        self.typeColumn = typeColumn
        self.company = company
        self.noneValue = noneValue

    def __extractType(self, row):
        if self.company not in str(row[self.companyColumn]):
            return self.noneValue

        else:
            S_Data = str(row[self.dataColumn])

            if S_Data != '' and S_Data != self.noneValue:
                S_Data_bin = str(bin(int(S_Data[:2], 16)))[2:].zfill(8)
                return hex(int(S_Data_bin[5:8], 2))[2:]

            else:
                return self.noneValue

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.typeColumn] = dataToProcess.apply(self.__extractType, axis=1)
        dataToProcess[self.typeColumn] = dataToProcess[self.typeColumn].astype("string")

        return dataToProcess


class dropLabels(ExecutorInterface):
    def __init__(self, labelColumn: str, labels: list[str]):
        self.labelColumn = labelColumn
        self.labels = labels

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess = dataToProcess[~dataToProcess[self.labelColumn].isin(self.labels)]
        return dataToProcess


class cleanPDU(ExecutorInterface):
    def __init__(self, column: str):
        self.column = column

    def __cleanPDU(self, x):
        if "Malformed Packet" in str(x):
            return str(x)[:-18]
        return str(x)

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.column] = dataToProcess[self.column].apply(self.__cleanPDU)
        return dataToProcess


class customColumnFunction(ExecutorInterface):
    def __init__(self, column: str, function: Callable):
        self.column = column
        self.function = function

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess[self.column] = dataToProcess[self.column].apply(self.function)
        return dataToProcess


class orderDataFrame(ExecutorInterface):
    def __init__(self, column: str, ascending: bool = True):
        self.column = column
        self.ascending = ascending

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        dataToProcess.sort_values(by=self.column, ascending=self.ascending, inplace=True)
        return dataToProcess


class createDummies(ExecutorInterface):
    def __init__(self,
                 column: str,
                 labels: dict[str:str] | list[str],
                 prefix: str,
                 splitting: bool = False,
                 NoneColumn: bool = False,
                 OtherColumn: bool = False,
                 OtherValue: str = 'Other',
                 NoneValue: str = '',
                 splitter: str = ',',
                 prefix_sep: str = ' ',
                 dropColumn: bool = False):

        self.column = column
        self.labels = labels
        self.prefix = prefix
        self.splitting = splitting
        self.NoneColumn = NoneColumn
        self.OtherColumn = OtherColumn
        self.OtherValue = OtherValue
        self.NoneValue = NoneValue
        self.splitter = splitter
        self.prefix_sep = prefix_sep
        self.dropColumn = dropColumn

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        def get_column(value, customPrefix=self.prefix, customSep=self.prefix_sep):
            return customPrefix + customSep + str(value)

        assert self.column in dataToProcess.columns
        temp = dataToProcess[[self.column]].copy(deep=True)

        labelList = []

        if isinstance(self.labels, dict):
            assert self.OtherValue not in self.labels.values()
            assert self.NoneValue not in self.labels.values()

            labelList = self.labels.values()

            for key in self.labels:
                temp[self.column] = temp[self.column].str.replace(key, self.labels[key])

        elif isinstance(self.labels, list):
            assert self.OtherValue not in self.labels
            assert self.NoneValue not in self.labels

            labelList = self.labels

        if self.splitting:
            temp[self.column] = temp[self.column].str.split(self.splitter, expand=False)
            temp = temp.explode(self.column)

        assert len(labelList) == len(set(labelList))

        finalColumns = [get_column(label) for label in labelList]

        if self.OtherColumn:
            assert self.OtherValue not in labelList
            temp.loc[(~temp[self.column].isin(labelList)) & (temp[self.column] != self.NoneValue), [
                self.column]] = self.OtherValue
            finalColumns.append(get_column(self.OtherValue))

        if self.NoneColumn:
            finalColumns.append(get_column(self.NoneValue))

        assert self.NoneValue not in labelList

        temp = pd.get_dummies(temp[self.column], prefix=self.prefix, prefix_sep=self.prefix_sep, dtype=int)

        for label in finalColumns:
            if label not in temp.columns:
                temp[label] = 0

        temp = temp[finalColumns]
        temp = temp.groupby(temp.index).sum().reset_index(drop=True)

        assert dataToProcess.shape[0] == temp.shape[0]
        dataToProcess = dataToProcess.join(temp, how='inner')

        if self.dropColumn:
            dataToProcess.drop(self.column, inplace=True, axis=1)

        return dataToProcess
