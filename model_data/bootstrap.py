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

from sklearn.utils import resample
import pickle

'''
Run a normal decision tree classifier. Out put the top 5 classifiers and the the features importance for all the 
features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runDecisionTreeClassifier(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the decision tree classifier
    clf = tree.DecisionTreeClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    dt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    dt_score = accuracy_score(y_test, dt_predictions)
    print("decision tree classification accuracy on test data is " + str(dt_score), file=sys.stderr)
    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)


'''
Function to run the gradient boosting classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runGradientBoostingClassifier(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the gradient boosting classifier
    clf = GradientBoostingClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    gbc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, gbc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)


'''
Function to run the adaboost classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runAdaBoostClassifier(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the adaboost classifier
    clf = AdaBoostClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    ada_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    dt_score = accuracy_score(y_test, ada_predictions)
    print("adaboost classification accuracy on test data is " + str(dt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)
    return (train_score, dt_score)


'''
Function to run the extra trees classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runExtraTreesClassifier(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the extra tree classifier
    clf = ExtraTreesClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    et_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    et_score = accuracy_score(y_test, et_predictions)
    print("Extra Tree classification accuracy on test data is " + str(et_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)

'''
Function to run the support vector classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runSupportVec(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the support vector classifier
    clf = svm.SVC()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    svm_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    svm_score = accuracy_score(y_test, svm_predictions)
    print("SVM classification accuracy on test data is " + str(svm_score), file=sys.stderr)

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
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def runXGBoost(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the extra gradient boosting classifier
    clf = XGBClassifier()
    clf.set_params(**p)


    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    xgb_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    xgb_score = accuracy_score(y_test, xgb_predictions)
    print("XGB classification accuracy on test data is " + str(xgb_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score)

'''
Function to run the random forest classifier on the data given. Out put the top 5 classifiers and the the features 
importance for all the features being used

x_train - training features
y_train - training GR/ DN labels
x_test - testinf features
y_test - testing GR/ DN labels
p - best parameters from grid search

:return train accuracy and test accuracy
'''
def random_trees(x_train, y_train, x_test, y_test, p):

    # Here we instantiate the random tree classifier
    clf = RandomForestClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    rt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    rt_score = accuracy_score(y_test, rt_predictions)
    print("Random Tree classification accuracy on test data is " + str(rt_score), file=sys.stderr)

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
    params = sys.argv[4]

    p_dict = {}
    with open(params, 'rb') as handle:
        p_dict = pickle.load(handle)

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

    feature_map['subset'] = ['MotionJurisNumber', 'CaseMajorCode_T', 'CaseMajorCode_V','CaseMajorCode_A', 'CaseMajorCode_C', 'CaseMajorCode_M', 'CaseMajorCode_P',
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
    
    feature_subset = list(set(mapped_features) & set(defendent_strikes_onehot.columns.values))
    MRC_IDX = feature_subset.index('MotionResultCode_GR')

    # handle missing data
    defendent_strikes_onehot = defendent_strikes_onehot[feature_subset]
    defendent_strikes_onehot = defendent_strikes_onehot.dropna()

    # Start bootstrap loop
    n_iterations = 100
    n_size = int(len(defendent_strikes_onehot) * 0.50)

    file_name = data.lower()
    x = ''
    if '_dm_' in data or '_dm.' in data:
        x = 'BBE'
    elif ('_dm_' not in data and '_dm.' not in data):
        x = 'og'

    rule = ''
    if 'simp' in file_name:
        rule = 'simp'
    elif 'foil' in file_name:
        rule = 'foil'
    else:
        rule = ''

    # drop outcome from features
    feature_subset.remove('MotionResultCode_GR')

    # 100 iterations of the boot strap
    rows = defendent_strikes_onehot.values
    for i in range(n_iterations):

        # 50/50 split during bootstrapping
        train, test = train_test_split(defendent_strikes_onehot, test_size=0.5,
                                       stratify=defendent_strikes_onehot.MotionResultCode_GR)


        x_train = train[feature_subset]
        y_train = train['MotionResultCode_GR']
        x_test = test[feature_subset]
        y_test = test['MotionResultCode_GR']

        import ast
        if alg == 0:
            print("decision tree classifier for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('decision tree',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('decision tree',fs,x)])
            except:
                jj = {}

                if rule != '':
                    jj = p_dict[('decision tree',fs,rule,x)]
                else:
                    jj = p_dict[('decision tree',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runDecisionTreeClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['decision tree', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 1:
            print("gradient boosting classifier for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('gradient boosting',fs,rule,x,)])
                else:
                    p = ast.literal_eval(p_dict[('gradient boosting',fs,x)])
            except:
                jj = {}

                if rule != '':
                    jj = p_dict[('gradient boosting',fs,rule,x,)]
                else:
                    jj = p_dict[('gradient boosting',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runGradientBoostingClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['gradient boosting', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 2:
            print("extra trees classifier for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('extra trees',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('extra trees',fs,x)])
            except:
                jj = {}

                if rule != '':
                    jj = p_dict[('extra trees',fs,rule,x)]
                else:
                    jj = p_dict[('extra trees',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runExtraTreesClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['extra trees', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 3:
            print("adaboost classifier for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('adaboost',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('adaboost',fs,x)])
            except:
                jj = {}

                if rule != '':
                    jj = p_dict[('adaboost',fs,rule,x)]
                else:
                    jj = p_dict[('adaboost',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random',max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runAdaBoostClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['adaboost', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 4:
            print("SVM for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('svm',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('svm',fs,x)])
            except:
                jj = {}

                if rule != '':
                    jj = p_dict[('svm',fs,rule,x)]
                else:
                    jj = p_dict[('svm',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runSupportVec(x_train, y_train, x_test, y_test,p)
            print("\t".join(['svm', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 5:
            print("XGB for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('xgb',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('xgb',fs,x)])
            except:
                jj = {}
                if rule != '':
                    jj = p_dict[('xgb',fs,rule,x)]
                else:
                    jj = p_dict[('xgb',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = runXGBoost(x_train, y_train, x_test, y_test,p)
            print("\t".join(['xgb', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
        elif alg == 6:
            print("random forest for " + data, file=sys.stderr)

            # parsing the best parameters from the parameter dictionary
            p = {}
            try:
                if rule != '':
                    p = ast.literal_eval(p_dict[('random forest',fs,rule,x)])
                else:
                    p = ast.literal_eval(p_dict[('random forest',fs,x)])
            except:
                jj = {}
                if rule != '':
                    jj = p_dict[('random forest',fs,rule,x)]
                else:
                    jj = p_dict[('random forest',fs,x)]
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test) = random_trees(x_train, y_train, x_test, y_test,p)
            print("\t".join(['random forest', data, str(accuracy_train), str(accuracy_test), fs]))
            print("\n", file=sys.stderr)
