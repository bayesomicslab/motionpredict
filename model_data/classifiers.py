''' scikit-learn is a Python library for Machine Learning.
    We will make use of their implementation of a decision tree classifier
    https://scikit-learn.org/stable/modules/tree.html,
    a function that helps us split data into training and test, and
    a function that helps us evaluate the accuracy of our model'''
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

'''
Run a normal decision tree classifier. Out put the top 5 classifiers and the the features importance for all the 
features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runDecisionTreeClassifier(x_train,y_train,x_test,y_test):

    # Here we instantiate the decision tree classifier
    clf = tree.DecisionTreeClassifier()

    parameter_grid = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'min_samples_split': [3, 4], # reg
                  'max_depth': [1, 2, 3, 4], # regularize
                  'min_samples_leaf': [2,3], # reg
                  'max_features': list(range(1,len(x_train.columns)+1))
                      }
    grid_search = GridSearchCV(clf, param_grid=parameter_grid, return_train_score=True,  n_jobs=20, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    # now, make the predictions using our classifier
    dt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    dt_score = accuracy_score(y_test, dt_predictions)
    print("decision tree classification accuracy on test data is " + str(dt_score), file=sys.stderr)

    # let's compare it to a naive algorithm that
    # simply selects the order code at random
    random_sample = np.random.choice([0,1], len(y_test), replace=True)
    r_score = accuracy_score(y_test, random_sample)
    print("random algorithm classification accuracy on test data is " + str(r_score), file=sys.stderr)

    print("feature importances " + str(clf.feature_importances_), file=sys.stderr)

    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names':[],'imp':[]}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp',ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f"%(i, row['names'],row['imp']), file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)


    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str( train_score ), file=sys.stderr)

    return (train_score,dt_score)

'''
Function to run the gradient boosting classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runGradientBoostingClassifier(x_train,y_train,x_test,y_test):

    # Here we instantiate the gradient boosting classifier
    clf = GradientBoostingClassifier()

    parameter_grid = {
        "learning_rate": np.linspace(0.001, .01, 3),# best below .1
        "min_samples_split": np.linspace(.3, .5, 2),  # original bot .1 to .5 with 12 - reg
        "min_samples_leaf": np.linspace(.3, .5, 2), # reg
        'max_depth': [1, 2, 3], # reg
        "subsample":[0.5, 0.6, 0.7], # reg
        "n_estimators":[100, 200, 300 ] # reg - larger slower to overfit
    }

    grid_search = GridSearchCV(clf, n_jobs=40, return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    gbc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, gbc_predictions)
    print("accuracy score on test data: " +str(dt_score), file=sys.stderr)
    print("feature importances " + str(clf.feature_importances_), file=sys.stderr)

    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names':[],'imp':[]}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp', ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f"%(i, row['names'],row['imp']),file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)



    etc_predictions = clf.predict(x_test)
    dt_score =accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str( train_score ), file=sys.stderr)

    return (train_score,dt_score)


'''
Function to run the adaboost classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runAdaBoostClassifier(x_train,y_train,x_test,y_test):

    # Here we instantiate the adaboost classifier
    clf = AdaBoostClassifier()

    parameter_grid = {
    "base_estimator": [tree.DecisionTreeClassifier(splitter='random',max_depth=1)], # lower depth and min sample split
    "learning_rate": np.linspace(0.0001, .001, 10),
    "n_estimators":[100,300,500]
    }

    grid_search = GridSearchCV(clf, n_jobs=20,  return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    print("feature importances " + str(clf.feature_importances_), file=sys.stderr)

    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names':[],'imp':[]}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp',ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f"%(i, row['names'],row['imp']),file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    etc_predictions = clf.predict(x_test)
    dt_score =accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str( train_score ), file=sys.stderr)
    return (train_score,dt_score)

'''
Function to run the extra trees classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runExtraTreesClassifier(x_train,y_train,x_test,y_test):

    # Here we instantiate the extra tree classifier
    clf = ExtraTreesClassifier()

    parameter_grid = {
        "criterion": ["gini",  "entropy"],
        "n_estimators":[100,150,200,250,300,400,500],
        "max_depth":[2,3,4], # reg
        "min_samples_split": np.linspace(.3, .5, 5), # original bot .1 to .5 with 12
        "min_samples_leaf": np.linspace(.3, .5, 5),
        "max_features": ["log2", "sqrt"]
    }

    grid_search = GridSearchCV(clf, n_jobs=20,  return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    importances = clf.feature_importances_
    print(importances, file=sys.stderr)
    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names':[],'imp':[]}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp',ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f"%(i, row['names'],row['imp']),file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    etc_predictions = clf.predict(x_test)
    dt_score =accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str( train_score ), file=sys.stderr)

    return (train_score,dt_score)

'''
Function to run the support vector classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runSupportVec(x_train,y_train,x_test,y_test):

    # Here we instantiate the support vector classifier
    clf = svm.SVC()
    parameter_grid = {'C': np.linspace(0.001,.1,100),
                        'gamma': [1, 0.1, 0.001, 0.0001],
                        'kernel': ['rbf']
                      }

    grid_search = GridSearchCV(clf, n_jobs=40, return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)
    
    return (train_score, dt_score)

'''
Function to run the extra gradient boosting classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def runXGBoost(x_train,y_train,x_test,y_test):

    parameter_grid = {
        'reg_lambda': np.linspace(27, 28 , 2),  # for over fitting
        'reg_alpha': np.linspace(27, 28, 2),
        "learning_rate": np.linspace(.0001, .001, 3), # usually between .05 and .3
        "max_depth": [2],
        "num_boosting_rounds": [1000],
        'nthread': [10]
    }

    # Here we instantiate the extra gradient boosting classifier
    clf = XGBClassifier()
    grid_search = GridSearchCV(clf, n_jobs=40, return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)

    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    importances = clf.feature_importances_
    print(importances, file=sys.stderr)

    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names': [], 'imp': []}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp', ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f" % (i, row['names'], row['imp']), file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)


'''
Function to run the random trees classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels

:return train accuracy and test accuracy
'''
def random_trees(x_train,y_train,x_test,y_test):

    # Here we instantiate the random forest classifier
    clf = RandomForestClassifier()
    parameter_grid = {"n_estimators": [100, 200, 250, 300, 400, 500],
                  "criterion": ["gini",  "entropy"],
                  "max_depth":[2,3,4,5],
                  "max_features": ["log2", "sqrt"],
                  "min_samples_split": np.linspace(0.3, 0.5, 10),
                  "min_samples_leaf": np.linspace(0.3, 0.5, 10),
                  'n_jobs': [10]
    }

    grid_search = GridSearchCV(clf, n_jobs=20,  return_train_score=True, param_grid=parameter_grid, cv=StratifiedKFold(n_splits=10))
    grid_search.fit(x_train,y_train)
    print('Best score: {}'.format(grid_search.best_score_), file=sys.stderr)
    print('Best parameters: {}'.format(grid_search.best_params_), file=sys.stderr)

    # refit and train the model to the best features and training data
    clf = grid_search.best_estimator_

    importances = clf.feature_importances_
    print(importances, file=sys.stderr)
    # Print the feature ranking
    print("Feature ranking:", file=sys.stderr)
    importanceDict = {'names': [], 'imp': []}
    for name, importance in zip(x_train.columns, clf.feature_importances_):
        importanceDict['names'] += [name]
        importanceDict['imp'] += [importance]
    fRank = pd.DataFrame.from_dict(importanceDict)
    fRank = fRank.sort_values(by='imp', ascending=False)
    i = 0
    for index, row in fRank.iterrows():
        print("%d. %s %f" % (i, row['names'], row['imp']), file=sys.stderr)
        i += 1

    cv_results = pd.DataFrame(grid_search.cv_results_)[['rank_test_score', 'params','mean_test_score','mean_train_score']]
    sorted_results = cv_results.sort_values(by='rank_test_score').head(5)

    print("\nTop 5 best Parameters: ", file=sys.stderr)
    for index, row in sorted_results.iterrows():
        print("%d. %s train: %s test: %s" % (row['rank_test_score'], str(row['params']), str(row['mean_train_score']), str(row['mean_test_score'])), file=sys.stderr)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)

if __name__ == '__main__':

    # Command line arguments need to run the code
    # data = file
    # alg = classifier to run
    # fs = features to use
    data = sys.argv[1]
    alg = int(sys.argv[2])
    fs = sys.argv[3]

    # Features to train on full, minimal, subset
    feature_map = {}
    feature_map['full'] = ['CaseAttorneyJuris', 'CaseLocation_AAN',
                           'CaseLocation_DBD', 'CaseLocation_FBT', 'CaseLocation_FST',
                           'CaseLocation_HHB', 'CaseLocation_HHD', 'CaseLocation_KNL',
                           'CaseLocation_KNO', 'CaseLocation_LLI', 'CaseLocation_MMX',
                           'CaseLocation_NNH', 'CaseLocation_NNI', 'CaseLocation_TTD',
                           'CaseLocation_UWY', 'CaseLocation_WWM', 'CaseLocation_nan',
                           'CaseAttorneyType_A', 'CaseAttorneyType_G',
                           'CaseAttorneyType_J', 'CaseAttorneyType_K', 'CaseAttorneyType_M',
                           'CaseAttorneyType_Q', 'CaseAttorneyType_R', 'CaseAttorneyType_U',
                           'CaseAttorneyType_X', 'CaseAttorneyType_nan',
                           'CaseDispositionDocketLegendCode_ARBJDG',
                           'CaseDispositionDocketLegendCode_FFJDG',
                           'CaseDispositionDocketLegendCode_JDGACT',
                           'CaseDispositionDocketLegendCode_JDGACTD',
                           'CaseDispositionDocketLegendCode_JDGACTP',
                           'CaseDispositionDocketLegendCode_JDGDACT',
                           'CaseDispositionDocketLegendCode_JDGHD',
                           'CaseDispositionDocketLegendCode_JDGMENT',
                           'CaseDispositionDocketLegendCode_JDGNST',
                           'CaseDispositionDocketLegendCode_JDGRPT',
                           'CaseDispositionDocketLegendCode_JDGSTA',
                           'CaseDispositionDocketLegendCode_JDGSTAP',
                           'CaseDispositionDocketLegendCode_JDGSTD',
                           'CaseDispositionDocketLegendCode_JDGSTP',
                           'CaseDispositionDocketLegendCode_JDGSTPP',
                           'CaseDispositionDocketLegendCode_JDGTVT',
                           'CaseDispositionDocketLegendCode_JDGTVTD',
                           'CaseDispositionDocketLegendCode_JDGTVTP',
                           'CaseDispositionDocketLegendCode_JDJVTCT',
                           'CaseDispositionDocketLegendCode_JGNSA',
                           'CaseDispositionDocketLegendCode_JGVMAPC',
                           'CaseDispositionDocketLegendCode_JOD',
                           'CaseDispositionDocketLegendCode_JODD',
                           'CaseDispositionDocketLegendCode_JWT',
                           'CaseDispositionDocketLegendCode_JWTD',
                           'CaseDispositionDocketLegendCode_JWTP',
                           'CaseDispositionDocketLegendCode_SATJFBS',
                           'CaseDispositionDocketLegendCode_SJ',
                           'CaseDispositionDocketLegendCode_SJD',
                           'CaseDispositionDocketLegendCode_SJP',
                           'CaseDispositionDocketLegendCode_TOUSDCT',
                           'CaseDispositionDocketLegendCode_TTFSTHS',
                           'CaseDispositionDocketLegendCode_WDACT',
                           'CaseDispositionDocketLegendCode_WDCWC',
                           'CaseDispositionDocketLegendCode_WDRAPJ',
                           'CaseDispositionDocketLegendCode_WOARD',
                           'CaseDispositionDocketLegendCode_nan',
                           'CaseDispositionJudgeJurisNo',
                           'CaseMajorCode_T', 'CaseMajorCode_V', 'CaseMajorCode_nan',
                           'CaseMarkingCode', 'CaseMinorCode', 'CaseReferenceNumber',
                           'CaseTrialListType', 'MotionJurisNumber',
                           'MotionTimeDuration', 'SelfRepBeforeMotionDecidedBool',
                           'SelfRepBeforeMotionDecidedCount', 'SelfRepBeforeMotionFileBool',
                           'SelfRepBeforeMotionFileCount',
                           "MotionResultCode_GR"]

    # trying smaller subsets of features
    feature_map['minimal'] = ['CaseAttorneyJuris',
                              'MotionTimeDuration',
                              "MotionResultCode_GR"]

    feature_map['subset'] = ['MotionJurisNumber', 'CaseMajorCode_T', 'CaseMajorCode_V',
                             'MotionTimeDuration', 'CaseLocation_AAN',
                             'CaseLocation_DBD', 'CaseLocation_FBT', 'CaseLocation_FST',
                             'CaseLocation_HHB', 'CaseLocation_HHD', 'CaseLocation_KNL',
                             'CaseLocation_KNO', 'CaseLocation_LLI', 'CaseLocation_MMX',
                             'CaseLocation_NNH', 'CaseLocation_NNI', 'CaseLocation_TTD',
                             'CaseLocation_UWY', 'CaseLocation_WWM', "MotionResultCode_GR"]

    # load in the data
    law_data = pd.read_csv(data, sep="\t")

    cols = list(law_data.columns)
    ef = []
    if 'SelfRepBeforeMotionDecidedBool' in cols:
        i = cols.index('SelfRepBeforeMotionDecidedBool') + 1
        ef = cols[i:]
    mapped_features = feature_map[fs] + ef

    # ** potential new approach for paper no time features **
    mapped_features = set(mapped_features) - set(['MotionTimeDuration'])

    # only use the motions filled by defendent
    defendent_strikes = law_data[law_data.MotionFilingParty == 'D']

    # filter out all but the case result codes we want
    defendent_strikes = defendent_strikes[(defendent_strikes.MotionResultCode == 'GR') |
                                          (defendent_strikes.MotionResultCode == 'DN') |
                                          (defendent_strikes.MotionResultCode == 'DS')]

    # one-hot encoding for categorical variables
    defendent_strikes_onehot = pd.get_dummies(defendent_strikes, dummy_na=True)

    # split into training and test data and stratify along predicting column
    train, test = train_test_split(defendent_strikes_onehot, test_size=0.3,
                                   stratify=defendent_strikes_onehot.MotionResultCode_GR)
    feature_subset = list(set(mapped_features) & set(defendent_strikes_onehot.columns.values))

    # handle missing data
    defendent_strikes_onehot = defendent_strikes_onehot.dropna()
    train = train[feature_subset].dropna()
    test = test[feature_subset].dropna()

    # drop outcome from features
    feature_subset.remove('MotionResultCode_GR')

    # data to train on
    x_train = train[feature_subset]
    # and the population as the outcome (what we want to predict)
    y_train = train["MotionResultCode_GR"]

    # this is the test data, we do not train using this data
    x_test = test[feature_subset]
    y_test = test["MotionResultCode_GR"]

    # run the one classifer requested
    if alg == 0:
        print("decision tree classifier for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runDecisionTreeClassifier(x_train, y_train, x_test, y_test)
        print("\t".join(['decision tree', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 1:
        print("gradient boosting classifier for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runGradientBoostingClassifier(x_train, y_train, x_test, y_test)
        print("\t".join(['gradient boosting', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 2:
        print("extra trees classifier for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runExtraTreesClassifier(x_train, y_train, x_test, y_test)
        print("\t".join(['extra trees', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 3:
        print("adaboost classifier for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runAdaBoostClassifier(x_train, y_train, x_test, y_test)
        print("\t".join(['adaboost', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 4:
        print("SVM for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runSupportVec(x_train, y_train, x_test, y_test)
        print("\t".join(['svm', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 5:
        print("XGB for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = runXGBoost(x_train, y_train, x_test, y_test)
        print("\t".join(['xgb', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
    elif alg == 6:
        print("random forest for " + data, file=sys.stderr)
        (accuracy_train, accuracy_test) = random_trees(x_train, y_train, x_test, y_test)
        print("\t".join(['random forest', data, str(accuracy_train), str(accuracy_test), fs]))
        print("\n", file=sys.stderr)
