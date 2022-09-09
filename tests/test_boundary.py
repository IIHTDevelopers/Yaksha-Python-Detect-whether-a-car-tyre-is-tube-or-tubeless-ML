#import os
import unittest
#import pickle
import numpy as np
from code.ml import Model
model = Model()
#file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_boundary_revised.txt'
from tests.TestUtils import TestUtils
class BoundaryTests(unittest.TestCase):
    def test_is_model_underfitting(self):
        test_obj = TestUtils()
        try:
        # this is for classification
            X_train, X_test, y_train, y_test = model.data_transformation()
            max_occuring_label = np.bincount(y_train).argmax()
            predictions = model.model_predict(X_test)

            benchmark_acc = model.cost_metric(
                y_true=y_test, y_pred=[max_occuring_label]*y_test.shape[0]
            )

            predicted_acc = model.cost_metric(
                y_true=y_test, y_pred=predictions
            )

            if predicted_acc > benchmark_acc:
                passed = True
                test_obj.yakshaAssert("TestIsModelUnderfitting",True,"boundary")
                print("TestIsModelUnderfitting = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestIsModelUnderfitting",False,"boundary")
                print("TestIsModelUnderfitting= Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestIsModelUnderfitting",False,"boundary")
            print("TestIsModelUnderfitting = Failed")
        assert passed

    def test_is_model_overfitting(self):
        test_obj = TestUtils()
        try:
            X_train, X_test, y_train, y_test = model.data_transformation()

            train_predict = model.model_predict(X_train)
            train_acc = model.cost_metric(
                y_true=y_train.values, y_pred=train_predict
            )

            test_predict = model.model_predict(X_test)
            test_acc = model.cost_metric(
                y_true=y_test.values, y_pred=test_predict
            )

            perc_10 = (train_acc/100)*10

            diff = abs(train_acc-test_acc)

            if diff < perc_10:
                passed = True
                test_obj.yakshaAssert("TestIsModelOverfitting",True,"boundary")
                print("TestIsModelOverfitting = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestIsModelOverfitting",False,"boundary")
                print("TestIsModelOverfitting = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestIsModelOverfitting",False,"boundary")
            print("TestIsModelOverfitting = Failed")
        assert passed
