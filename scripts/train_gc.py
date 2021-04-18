#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pickle
import time
import argparse
import gc as gc_m
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from scripts.gcforest.gcforest import GCForest
from scripts.utils import check_path

__doc__ = """
train_gc - a training, testing scripts for gcforest module
==========================================================

**train_gc** is a Python script for gcforest module training, testing.

Main Functions
--------------
Here are just a few of the things that **train_gc** does well:

  - Configuration and training of GCForest modules.

Main Program Functions
----------------------

"""

def get_ca_config():
    '''Get the structure of gcForest cascading modules.

    Returns:

        Configuration information.
    '''
    config = {}
    ca_config = {}
    ca_config["random_state"] = 777
    ca_config["max_layers"] = 7
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 15, "max_depth": 7, "num_class": 2,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": 5, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": 5, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 10, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def run_gc(X_train, X_test, y_train, gc_model_path):
    '''Train gcForest models, predict results and save models.

    Args:

        X_train: The training set.

        X_test: The testing set

        y_train: The training set label.

        gc_model_path: Model saving path.

    Returns:
        The prediction of y.
    '''
    config = get_ca_config()
    gc = GCForest(config)
    gc.set_keep_model_in_mem(True)
    gc.set_keep_data_in_mem(False)

    gc.fit_transform(X_train, y_train)
    with open(gc_model_path, "wb") as f:
        pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    y_pred = gc.predict(X_test)
    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_gc:')

    args = parser.parse_args()

    print("Running...")

    result_gc = []

    gc_time = 0

    for i in range(10):
        re_data_path = '../features/profile/re_data/'
        train_path = re_data_path + 'Xc_train_' + str(i) + '.pkl'
        test_path = re_data_path + 'Xc_test_' + str(i) + '.pkl'
        ytr_path = re_data_path + 'y_train_' + str(i) + '.pkl'
        yte_path = re_data_path + 'y_test_' + str(i) + '.pkl'
        start_cnn = time.time()

        with open(train_path, 'rb') as f:
            Xc_train = pickle.load(f)
        with open(test_path, 'rb') as f:
            Xc_test = pickle.load(f)
        with open(ytr_path, 'rb') as f:
            y_train = pickle.load(f)
        with open(yte_path, 'rb') as f:
            y_test = pickle.load(f)

        gc_model_path = '../model/gc_' + str(i) + '.pkl'

        start_gc = time.time()
        y_pred = run_gc(Xc_train, Xc_test, y_train, gc_model_path)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        from sklearn.metrics import confusion_matrix

        c = confusion_matrix(y_test, y_pred)
        sp = (c[0][0]) / (c[0][0] + c[0][1])
        se = (c[1][1]) / (c[1][1] + c[1][0])
        result_gc.append([acc, f1, sp, se])
        gc_time += (time.time() - start_gc)

        del Xc_train, Xc_test, y_train, y_test
        gc_m.collect()

        i += 1
        break

    result_gc = pd.DataFrame(result_gc, columns=['acc', 'f1', 'sp', 'se'])
    result_gc.loc['mean'] = result_gc.apply(lambda x: x.mean())

    from scripts.gcforest.utils.log_utils import strftime

    tem_path = '../output/' + 'gc_model_train/'
    check_path(tem_path)

    result_gc.to_csv(tem_path + 'gc_' + str(gc_time) + '_result_' + strftime() + '.csv')
