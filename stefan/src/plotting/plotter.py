from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.pre_processing.interfaces.executor_interface import ExecutorInterface


class Plotter(ExecutorInterface, ABC):
    def __init__(self, columns: dict[str, str], figSize: tuple[float, float] = (16, 10), dpi: int = 300,
                 title: str = None, xLabel: str = None, yLabel: str = None, savePath: str = None, show: bool = True,
                 noneValue='None', otherValue='Other'):

        self.columns = columns
        self.dpi = dpi
        self.figSize = figSize
        self.savePath = savePath
        self.title = title
        self.yLabel = yLabel
        self.xLabel = xLabel
        self.show = show
        self.noneValue = noneValue
        self.otherValue = otherValue

    @abstractmethod
    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        pass

    def checkColumns(self, columns: list):
        for column in columns:
            assert column in self.columns

    def setUpFig(self):
        plt.figure(figsize=self.figSize, dpi=self.dpi)

    def showFig(self):
        if self.show:
            plt.show()

    def resetFig(self):
        plt.close()

    def teardown(self, title: str = None):
        plt.tight_layout()
        self.savePlot(title)
        self.showFig()
        self.resetFig()

    def savePlot(self, title=None):
        if title is None:
            title = self.title

        if not (self.savePath is None) and not (title is None):
            name = title
            path = self.savePath + name
            plt.savefig(path, dpi=self.dpi)

    def classLabels(self, dataset: pd.DataFrame, labelColumn: str):
        labels = dataset[labelColumn].unique()
        return sorted(list(labels))

    def selectCategories(self, dataToPlot: pd.DataFrame, featureColumn: str, labelColumn: str,
                         max_count: int = 9, includeNone: bool = True) -> list:

        sizes = (dataToPlot.groupby([labelColumn, featureColumn]).size() /
                 dataToPlot.groupby([labelColumn]).size()).reset_index().rename(columns={0: 'Size'})

        average_sizes = sizes.groupby([featureColumn])['Size'].mean().reset_index().rename(columns={'Size': 'avg Size'})
        average_sizes = average_sizes.sort_values(by='avg Size', ascending=False)

        temp = average_sizes.iloc[:max_count, :]
        temp_features = list(temp[featureColumn].unique())

        if self.noneValue not in temp_features and includeNone:
            temp = average_sizes.iloc[:max_count - 1, :]
            temp_features = list(temp[featureColumn].unique()) + [self.noneValue]

        elif self.noneValue in temp_features and not includeNone:
            temp = average_sizes.iloc[:max_count + 1, :]
            temp_features = list(temp[featureColumn].unique())
            temp_features.remove(self.noneValue)

        return temp_features

    def plotAxisLabelsAndTitle(self, plotAxis: bool = True, plotTitle: bool = True):
        if plotAxis:
            if not (self.xLabel is None):
                plt.xlabel(self.xLabel, fontsize=15)
            if not (self.yLabel is None):
                plt.ylabel(self.yLabel, fontsize=15)
        if plotTitle:
            if not (self.title is None):
                plt.title(self.title, fontsize=20)

    def plotLegend(self):
        plt.legend(loc='upper right', fontsize=12)


class AbstractComplexBar(Plotter, ABC):
    def createBarPlotDf(self, dataToPlot: pd.DataFrame, featureColumn: str, categories: list = None,
                        labelColumn: str = 'Label', sizes: pd.DataFrame.groupby = None, sizeColumn: str = 'Size',
                        includeOther: bool = True, threshold: float = 0.002, ignoreNone: bool = False) -> pd.DataFrame:

        if categories is None:
            categories = list(dataToPlot[featureColumn].unique().tolist())

        def clean(x):
            if x in categories:
                return x
            elif ignoreNone and x == self.noneValue:
                return self.noneValue

            return self.otherValue

        dataToPlot[featureColumn] = dataToPlot[featureColumn].apply(clean)

        if sizes is None:
            df = (dataToPlot.groupby([labelColumn, featureColumn]).size() / dataToPlot.groupby(
                labelColumn).size()).reset_index().rename(columns={0: sizeColumn})
        else:
            df = (dataToPlot.groupby(
                [labelColumn, featureColumn]).size() / sizes).reset_index().rename(
                columns={0: sizeColumn})

        final_labels = sorted(list(set(df[labelColumn].unique())))

        df = df[df[sizeColumn] >= threshold]

        if not includeOther:
            df = df[df[featureColumn] != self.otherValue]

        if ignoreNone:
            df = df[df[featureColumn] != self.noneValue]

        final_categories = sorted(list(set(df[featureColumn].unique())))

        combinations = pd.DataFrame(list(product(final_labels, final_categories)), columns=[labelColumn, featureColumn])
        combinations = combinations.merge(df, on=[labelColumn, featureColumn], how='left').fillna(0)

        return combinations

    def plotMultiBar(self, dataToPlot: pd.DataFrame, featureColumn: str, labelColumn: str, sizeColumn: str = 'Size',
                     percent: bool = False):
        classLabels = list(dataToPlot[labelColumn].unique())
        bars = list(dataToPlot[featureColumn].unique())

        width = 1
        step_size_base = (len(bars) * width)
        step_size = step_size_base + width * 4
        x = np.array([i * step_size for i, e in enumerate(classLabels)])

        self.setUpFig()

        for index, bar in enumerate(bars):
            x_new = x - step_size_base / 2 + index * width
            plt.bar(x_new, dataToPlot[dataToPlot[featureColumn] == bar][sizeColumn], label=bar, width=width,
                    align='edge')

        plt.xticks(x, classLabels, rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        if percent:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.ylim(0)

        self.plotAxisLabelsAndTitle()

        self.plotLegend()
        self.teardown()

    def plotStackedBar(self, dataToPlot: pd.DataFrame, featureColumn: str, labelColumn: str, sizeColumn: str = 'Size'):
        dataToPlot[sizeColumn] = 100 * dataToPlot[sizeColumn]

        classes = list(dataToPlot[labelColumn].unique())
        bars = list(dataToPlot[featureColumn].unique())

        y_prev = np.array([0.0 for i in classes])
        self.setUpFig()

        for bar in bars:
            y = dataToPlot[dataToPlot[featureColumn] == bar][sizeColumn].to_numpy()
            plt.bar(classes, y, bottom=y_prev, label=bar)
            y_prev += y

        plt.gca().yaxis.set_major_formatter(PercentFormatter())
        plt.ylim(bottom=0, top=110)
        plt.xticks(rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        self.plotAxisLabelsAndTitle()
        self.plotLegend()
        self.teardown()


class AbstractSimpleBar(Plotter, ABC):
    def addLabels(self, x, y):
        offset = y.max() * 0.01
        for i in range(len(x)):
            plt.text(i, y[i] + offset, str(int(np.round(y[i]))), ha='center', fontsize=12)

    def plotSimpleBar(self, x: pd.Series, y: pd.Series, log=False, percent=False, showNumbers=False):
        self.setUpFig()

        plt.bar(x, y)
        plt.xticks(rotation=90, fontsize=15)
        plt.yticks(fontsize=15)

        if log:
            plt.yscale("log")

        if showNumbers:
            self.addLabels(x, y)

        self.plotAxisLabelsAndTitle()

        plt.ylim(0, top=y.max() * 1.1)

        if percent:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        self.teardown()


class AbstractBoxplot(Plotter, ABC):
    def plotBoxPlot(self, dataToPlot: pd.DataFrame, featureColumn: str, labelColumn: str):
        classLabels = self.classLabels(dataToPlot, labelColumn)

        boxplot = [dataToPlot[dataToPlot[labelColumn] == label][featureColumn].to_numpy() for label in
                   classLabels]

        self.setUpFig()
        plt.boxplot(boxplot, labels=classLabels, showfliers=True, whis=1)
        plt.xticks(rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(0)

        self.plotAxisLabelsAndTitle()

        self.teardown()


class StackedBarPlot(AbstractComplexBar):
    requiredColumns = ['Label', 'Feature']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        categories = self.selectCategories(dataToProcess, featureColumn=self.columns['Feature'],
                                           labelColumn=self.columns['Label'])
        temp = self.createBarPlotDf(dataToProcess, featureColumn=self.columns['Feature'],
                                    labelColumn=self.columns['Label'],
                                    categories=categories)
        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotStackedBar(temp, featureColumn=self.columns['Feature'], labelColumn=self.columns['Label'])

        return dataToProcess


class BoxPlot(AbstractBoxplot):
    requiredColumns = ['Label', 'Feature']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        return dataToProcess

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)
        self.plotBoxPlot(temp, featureColumn=self.columns['Feature'], labelColumn=self.columns['Label'])

        return dataToProcess


class MeanValuesPlot(AbstractSimpleBar):
    requiredColumns = ['Label', 'Feature']

    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        labelColumn = self.columns['Label']
        featureColumn = self.columns['Feature']

        labels = self.classLabels(dataToProcess, labelColumn)

        dataToProcess = dataToProcess[dataToProcess[featureColumn] > 0]
        temp = dataToProcess.groupby(labelColumn)[featureColumn].mean().reset_index()

        combinations = pd.DataFrame(labels, columns=[labelColumn])
        temp = combinations.merge(temp, on=[labelColumn], how='left').fillna(0)

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)
        self.plotSimpleBar(temp[self.columns['Label']], temp[self.columns['Feature']], showNumbers=True)
        return dataToProcess


class PlotNumberOfPackets(AbstractSimpleBar):
    requiredColumns = ['Label']

    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        return dataToProcess.groupby(self.columns['Label']).size().reset_index().rename(columns={0: 'Count'})

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotSimpleBar(temp['Label'], temp['Count'], showNumbers=True)

        return dataToProcess


class PlotAddressIntervalBoxPlot(AbstractBoxplot):
    requiredColumns = ['Label', 'Source', 'Time']

    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        temp = (dataToProcess.groupby([self.columns['Label'], self.columns['Source']])[self.columns['Time']].max()
                - dataToProcess.groupby([self.columns['Label'], self.columns['Source']])[
                    self.columns['Time']].min()).reset_index()

        temp[self.columns['Time']] = temp[self.columns['Time']].dt.total_seconds() / 60

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)
        self.plotBoxPlot(temp, featureColumn=self.columns['Time'], labelColumn=self.columns['Label'])

        return dataToProcess


class PlotAddressIntervalMean(AbstractSimpleBar):
    requiredColumns = ['Label', 'Source', 'Time']

    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        temp = (dataToProcess.groupby([self.columns['Label'], self.columns['Source']])[self.columns['Time']].max()
                - dataToProcess.groupby([self.columns['Label'], self.columns['Source']])[
                    self.columns['Time']].min()).reset_index()

        temp[self.columns['Time']] = temp[self.columns['Time']].dt.total_seconds() / 60

        return temp.groupby(self.columns['Label'])[self.columns['Time']].mean().reset_index()

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)
        self.plotSimpleBar(temp['Label'], temp['Time'], showNumbers=True)

        return dataToProcess


class PlotBroadcast(AbstractSimpleBar):
    requiredColumns = ['Label', 'Broadcast']

    def createDataFrame(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        temp = (dataToProcess.groupby(self.columns['Label'])[self.columns['Broadcast']].sum()
                / dataToProcess.groupby(self.columns['Label']).size()).reset_index().rename(columns={0: 'Size'})

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotSimpleBar(temp[self.columns['Label']], temp['Size'], percent=True)

        return dataToProcess


class PlotProtocol(AbstractComplexBar):
    requiredColumns = ['Label', 'Protocol']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        categories = self.selectCategories(dataToProcess, featureColumn=self.columns['Protocol'],
                                           labelColumn=self.columns['Label'])
        temp = self.createBarPlotDf(dataToProcess, featureColumn=self.columns['Protocol'],
                                    labelColumn=self.columns['Label'], categories=categories)
        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotMultiBar(temp, featureColumn=self.columns['Protocol'], labelColumn=self.columns['Label'], percent=True)

        return dataToProcess


class PlotAdvertisementType(AbstractComplexBar):
    requiredColumns = ['Label', 'AD Type']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        original = dataToProcess.copy(deep=True)
        temp = dataToProcess.copy(deep=True)

        temp[self.columns['AD Type']] = temp[self.columns['AD Type']].str.split(',', expand=False)
        temp = temp.explode(self.columns['AD Type'])
        categories = self.selectCategories(temp, featureColumn=self.columns['AD Type'],
                                           labelColumn=self.columns['Label'], includeNone=False)

        df = self.createBarPlotDf(temp, featureColumn=self.columns['AD Type'], labelColumn=self.columns['Label'],
                                  categories=categories,
                                  sizes=original.groupby(self.columns['Label']).size(), ignoreNone=True)

        return df

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotMultiBar(temp, featureColumn=self.columns['AD Type'], labelColumn=self.columns['Label'], percent=True)
        return dataToProcess


class PlotUUID(AbstractComplexBar):
    requiredColumns = ['Label', 'UUID']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        original = dataToProcess.copy(deep=True)
        temp = dataToProcess.copy(deep=True)

        temp[self.columns['UUID']] = temp[self.columns['UUID']].str.split(',', expand=False)
        temp = temp.explode(self.columns['UUID'])
        categories = self.selectCategories(temp, featureColumn=self.columns['UUID'], labelColumn=self.columns['Label'],
                                           includeNone=False)

        df = self.createBarPlotDf(temp, featureColumn=self.columns['UUID'], labelColumn=self.columns['Label'],
                                  categories=categories,
                                  sizes=original.groupby(self.columns['Label']).size(), ignoreNone=True)

        return df

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotMultiBar(temp, featureColumn=self.columns['UUID'], labelColumn=self.columns['Label'], percent=False)
        return dataToProcess


class AbstractPacketRate(Plotter, ABC):
    sec = 60
    valid_channels = [37, 38, 39]


class PlotPacketRateMean(AbstractComplexBar, AbstractPacketRate):
    requiredColumns = ['File', 'Label', 'Channel', 'Time']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        temp = dataToProcess.copy(deep=True)
        temp = temp[temp[self.columns['Channel']].isin(self.valid_channels)]

        temp = temp.set_index(self.columns['Time'])

        temp = (temp.groupby([self.columns['File'], self.columns['Label'], self.columns['Channel']]).resample(
            str(self.sec) + "s").size() / self.sec).reset_index().rename(
            columns={0: "Rate"})

        temp.drop(self.columns['File'], axis=1, inplace=True)

        channels = sorted(list(temp[self.columns['Channel']].unique()))

        labels = self.classLabels(temp, self.columns['Label'])

        labelColumn = self.columns['Label']
        featureColumn = self.columns['Channel']

        temp = temp.groupby([labelColumn, featureColumn])['Rate'].mean().reset_index()

        combinations = pd.DataFrame(list(product(labels, channels)), columns=[labelColumn, featureColumn])
        temp = combinations.merge(temp, on=[labelColumn, featureColumn], how='left').fillna(0)

        temp[featureColumn] = temp[featureColumn].astype("string")
        temp[featureColumn] = "CH " + temp[featureColumn]

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)

        temp = self.createDataFrame(dataToProcess)

        self.plotMultiBar(temp, featureColumn=self.columns['Channel'], sizeColumn='Rate', percent=False,
                          labelColumn=self.columns['Label'])
        return dataToProcess


class PlotPacketRateBoxPlot(AbstractBoxplot, AbstractPacketRate):
    requiredColumns = ['File', 'Label', 'Channel', 'Time']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        temp = dataToProcess.copy(deep=True)
        temp = temp[temp[self.columns['Channel']].isin(self.valid_channels)]
        temp = temp.set_index(self.columns['Time'])

        sec = self.sec

        temp = (temp.groupby([self.columns['File'], self.columns['Label'], self.columns['Channel']]).resample(
            str(sec) + "s", include_groups=False).size() / sec).reset_index().rename(
            columns={0: "Rate"})

        temp[self.columns['Label']] = temp[self.columns['Label']] + " CH " + temp[self.columns['Channel']].astype("str")

        temp.drop([self.columns['File'], self.columns['Time'], self.columns['Channel']], axis=1, inplace=True)
        temp = temp.reset_index(drop=True)

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotBoxPlot(temp, featureColumn='Rate', labelColumn=self.columns['Label'])

        return dataToProcess


class PlotPacketRateGraph(AbstractPacketRate):
    requiredColumns = ['File', 'Label', 'Channel', 'Time']

    def plotPacketRate(self, df):
        channelColumn = self.columns['Channel']
        timeColumn = self.columns['Time']
        fileColumn = self.columns['File']
        labelColumn = self.columns['Label']

        file = df[fileColumn].iloc[0]
        label = df[labelColumn].iloc[0]

        self.setUpFig()

        def printLine(df2):
            channel = df2[channelColumn].iloc[0]
            x = df2[timeColumn][1:-1]
            y = df2['Rate'][1:-1]

            plt.plot(x, y, label="CH " + str(channel))

        df.groupby(channelColumn).apply(printLine)

        offset = 0.01
        max = df['Rate'].quantile(1 - offset)
        min = df['Rate'].quantile(offset)

        self.plotLegend()

        if self.title is None:
            title = file
            plt.title(title)

        else:
            title = self.title + " - " + label + " - (" + str(file) + ")"
            plt.title(title, fontsize = 20)

        self.plotAxisLabelsAndTitle(plotTitle=False)

        plt.ylim(bottom=min * 0.6, top=max * 1.4)
        plt.xlim(left=0)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        self.teardown(title)

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        channelColumn = self.columns['Channel']
        fileColumn = self.columns['File']
        labelColumn = self.columns['Label']
        timeColumn = self.columns['Time']

        temp = dataToProcess.copy(deep=True)
        temp = temp[temp[self.columns['Channel']].isin(self.valid_channels)]
        temp = temp.set_index(timeColumn)

        temp = (temp.groupby([fileColumn, labelColumn, channelColumn])
                .resample(str(self.sec) + "s").size() / self.sec).reset_index().rename(columns={0: "Rate"})

        temp[timeColumn] = temp[timeColumn] - temp.groupby([fileColumn])[timeColumn].transform('min')
        temp[timeColumn] = temp[timeColumn].dt.total_seconds() / 3600

        return temp

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        temp.groupby([self.columns['File'], self.columns['Label']]).apply(self.plotPacketRate)

        return dataToProcess


class PlotMalformedPacket(AbstractComplexBar):
    requiredColumns = ['Malformed', 'Label']

    def createDataFrame(self, dataToProcess) -> pd.DataFrame:
        dataToProcess[self.columns['Malformed']] = dataToProcess[self.columns['Malformed']].astype("bool")

        return self.createBarPlotDf(dataToProcess, featureColumn=self.columns['Malformed'], categories=[True, False],
                                    labelColumn=self.columns['Label'])

    def execute(self, dataToProcess: pd.DataFrame) -> pd.DataFrame:
        self.checkColumns(self.requiredColumns)
        temp = self.createDataFrame(dataToProcess)

        self.plotStackedBar(temp, featureColumn=self.columns['Malformed'], labelColumn=self.columns['Label'])
        return dataToProcess
