import unittest
import pandas as pd
from src.plotting.functions import featureImportanceDataFrame


class Test_Helpers(unittest.TestCase):
    def verifyResult(self, x: pd.DataFrame, y: pd.DataFrame):
        self.assertEqual(list(x.columns), list(y.columns))

        for column in x.columns:
            self.assertEqual(x[column].to_numpy().tolist(), y[column].to_numpy().tolist())

    def test_feature_importance_simple(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=False)

        features_result = ['F2', 'F3', 'F4', 'F1']
        importance_result = [40.0, 30.0, 20.0, 10.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(result, solution)

    def test_feature_importance_otherFeatures_simple(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=True, abs_limit=5, rel_limit=0)

        features_result = ['F2', 'F3', 'F4', 'F1']
        importance_result = [40.0, 30.0, 20.0, 10.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(solution, result)

    def test_feature_importance_otherFeatures_abs_limit(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=True, abs_limit=3, rel_limit=0)

        features_result = ['F2', 'F3', 'Other Features']
        importance_result = [40.0, 30.0, 30.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(solution, result)

    def test_feature_importance_otherFeatures_rel_limit(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=True, abs_limit=10, rel_limit=25)

        features_result = ['F2', 'F3', 'Other Features']
        importance_result = [40.0, 30.0, 30.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(solution, result)

    def test_feature_importance_otherFeatures_abs_limit_and_rel_limit_V1(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=True, abs_limit=2, rel_limit=15)

        features_result = ['Other Features', 'F2']
        importance_result = [60.0, 40.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(solution, result)

    def test_feature_importance_otherFeatures_abs_limit_and_rel_limit_V2(self):
        features = ['F1', 'F2', 'F3', 'F4']
        importance = [0.1, 0.4, 0.3, 0.2]

        result = featureImportanceDataFrame(features, importance, other=True, abs_limit=3, rel_limit=35)

        features_result = ['Other Features', 'F2']
        importance_result = [60.0, 40.0]
        solution = pd.DataFrame({'Feature': features_result, 'Importance': importance_result})

        self.verifyResult(solution, result)
