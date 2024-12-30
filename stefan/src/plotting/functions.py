import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

fontsize_label = 15
fontsize_ticks = 15


def plotReport(y_test, y_test_pred, labels, show=True, name: str = None, savePath: str = None, digits: int = 4):
    report = classification_report(y_test, y_test_pred, labels=labels, digits=digits, zero_division=0.0)

    prefix = "Classification Report"

    if show:
        if name is None:
            print()
        else:
            print(prefix + " - " + name)

        print(report)
        print("\n")

    if savePath is not None and name is not None:
        report = classification_report(y_test, y_test_pred, labels=labels, digits=digits, zero_division=0.0,
                                       output_dict=True)

        report = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Label'}).round(3)

        report_lower = report.iloc[:-3, :]
        report_empty = pd.DataFrame([['' for i in report.columns]], columns=report.columns)
        report_upper = report.iloc[-3:, ]

        report = pd.concat([report_lower, report_empty, report_upper])

        report.to_csv(savePath + prefix + " - " + name + ".csv", encoding='utf-8', index=False)


def plotTree(model, name: str, features_names: list[str], savePath: str = None, sizeFactor=1, dpi: int = 100,
             show=True):
    fig, ax = plt.subplots(figsize=(16 * sizeFactor, 9 * sizeFactor), dpi=dpi)
    plt.suptitle(name, fontsize=25 * sizeFactor)
    fig.tight_layout()

    tree.plot_tree(model, feature_names=features_names, class_names=model.classes_, ax=ax, filled=True)

    if savePath is not None:
        fig.savefig(savePath + name, dpi=500)

    if show:
        plt.show()

    plt.close()


def plotMatrix(y_test, y_test_pred, labels, dpi: int = 300, normalize: str = 'true', savePath: str = None,
               name: str = None, show: bool = True, digits: int = 2):
    cm = confusion_matrix(y_test, y_test_pred, labels=labels, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=dpi)

    prefix = "Confusion Matrix"

    if name is None:
        ax.set_title(prefix, fontsize=20)
    else:
        ax.set_title(prefix + " - " + name, fontsize=20)

    cm_display.plot(cmap='Blues', ax=ax, values_format=f".{digits}%", xticks_rotation=90)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    plt.xlabel('Predicted Label', fontsize=fontsize_label)
    plt.ylabel('True Label', fontsize=fontsize_label)

    fig.tight_layout()

    if name is not None and savePath is not None:
        fig.savefig(savePath + prefix + " - " + name, dpi=300)

    if show:
        plt.show()

    plt.close()


def featureImportanceDataFrame(features: np.array, importance: np.array, other: bool = False, rel_limit: float = 1,
                               abs_limit: int = 10):
    assert len(features) == len(importance)
    assert len(features) > 0

    frequency = pd.DataFrame(features, columns=['Feature'])
    frequency['Importance'] = importance
    frequency['Importance'] *= 100

    frequency.sort_values(by='Importance', ascending=False, inplace=True)
    frequency.reset_index(inplace=True, drop=True)

    if other:
        otherName = 'Other Features'
        minFeatureImportance = frequency['Importance'].min()
        cutOffIndex = 0

        if len(importance) <= abs_limit and minFeatureImportance > rel_limit:
            return frequency

        elif len(importance) > abs_limit and minFeatureImportance > rel_limit:
            cutOffIndex = abs_limit - 1

        elif len(importance) <= abs_limit and minFeatureImportance <= rel_limit:
            cutOffIndex = frequency.index[frequency['Importance'] <= rel_limit][0]

        elif len(importance) > abs_limit and minFeatureImportance <= rel_limit:
            cutOffIndex = min(abs_limit - 1, frequency.index[frequency['Importance'] <= rel_limit][0])

        frequency.loc[cutOffIndex:, "Feature"] = otherName
        frequency = frequency.groupby("Feature").sum().sort_values(by='Importance', ascending=False).reset_index()

    return frequency


def plotFeatureImportance(features, importance, dpi: int = 300, rel_limit: float = 1, abs_limit: int = 10,
                          savePath: str = None, name: str = None, show: bool = True):
    frequency = featureImportanceDataFrame(np.array(features), np.array(importance), rel_limit=rel_limit, other=True,
                                           abs_limit=abs_limit)

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    bars = plt.barh(frequency['Feature'], frequency['Importance'])

    if 'Other Features' in list(frequency['Feature'].unique()):
        indexOther = frequency.index[frequency['Feature'] == 'Other Features'][0]
        bars[indexOther].set_color('m')

    plt.gca().invert_yaxis()

    plt.xlabel('relative Feature Importance', fontsize=fontsize_label)

    prefix = "Feature Importance"

    if name is None:
        plt.title(prefix, fontsize=20)
    else:
        plt.title(prefix + " - " + name, fontsize=20)

    plt.gca().xaxis.set_major_formatter(PercentFormatter())

    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    fig.tight_layout()

    if savePath is not None and name is not None:
        fig.savefig(savePath + prefix + " - " + name, dpi=300)

    if show:
        plt.show()

    plt.close()
