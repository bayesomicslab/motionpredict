#packages/libraries needed to run the code.
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import sys

'''
This function takes in, as parameters, testing data and the classifier csv file outputed
from the classifier.py script. This script will print out the classification accuracy.

Command to run script: python3 name_of_this_file.py your_testing_data your_classifier_csv_file
'''

def ca(testingData, classifier):
    '''
    Drops text column in the testing data. Replaces the motion result code in the testing data
    as binary: GR (granted) is 1 and DN (Denied) is 0. Drops all uneccessary columns, but keeps
    the document id and motion result code columns. Then uses the model "accuracy_score"
    from the scikit-learn module to compute and print the classification accuracy.
    '''

    GR_classifier = pd.read_csv(classifier, sep='\t')
    testingData = pd.read_csv(testingData, sep='\t')
    testingtemp = testingData
    print("making dataframes comparable")
    testingtemp.drop('text', axis=1, inplace=True)
    testingtemp.MotionResultCode.replace(['GR', 'DN'], [1.0, 0.0], inplace=True)
    GR_classifier.drop(['GR_index', 'GR_Rule', 'GR_score'], axis=1, inplace=True)
    GR_classifier['MRCode'] = 1.0
    merged = pd.merge(testingtemp, GR_classifier, how='left', on='docid')
    merged['MRCode'] = merged['MRCode'].fillna(0.0)
    merged.rename(columns={'MRCode': 'predictions'}, inplace=True)
    print("looping through both dataframes....")

    actualList = merged['MotionResultCode'].to_list()
    predictionList = merged['predictions'].to_list()

    ca = accuracy_score(actualList, predictionList)
    print(ca)

if __name__ == '__main__':
    #This allows paramters to be passed on the command line.
    testingData = sys.argv[1]
    classifier = sys.argv[2]
    ca(testingData, classifier)
