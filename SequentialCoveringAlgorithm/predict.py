#packages/libraries needed to run the code.
import pandas as pd
import re
import csv
import sys

'''
This function takes training data as a parameter and the rules csv file as the second parameter that was
outputed from the sequential covering algorithm script. It then outputs a csv file containing all the documents
that were classified as granted or denied. The outputed csv file is called Foil_classifier_.csv or Simple_classifier_.csv depending on the 3rd parameter of the function.
It has 5 colunns: index of the rule, the rule, the Foil Information Gain function statistic for that rule, document id, and
motion result code as binary.

Command to run the script: python3 name_of_this_file.py your_testing_data your_rules type
type is simple or foil
'''
def predict(testingData, rules, type):
    print("reading in csv files ...")
    '''
    Reads in the tab delimited testing data and one hot encodes motion result code. GR (granted) is 1
    and DN (denied) is 0. Also reads in the csv file that contains the rules and predicts.
    After this process, it then turns the list of lists into the dataframe
    with 5 columns as explained above.
    '''
    GR_DN_nlp_testing = pd.read_csv(testingData, sep='\t')
    GR_DN_nlp_testing.MotionResultCode.replace(['GR', 'DN'], [1, 0], inplace=True)
    trainingRulesGR = pd.read_csv(rules, sep='\t')
    print("classifying GR....")
    GR_list = []
    testing_GR = GR_DN_nlp_testing
    for i, r in trainingRulesGR.iterrows():
        for index, row in testing_GR.iterrows():
            idCase = row[0]
            MRCode = row[2]
            listTX = row[1]
            if r[0] in listTX:
                GR_list.append([i, r[0], r[4], idCase, MRCode])
                testing_GR.drop(index, inplace=True)

    print("job done...")

    classificationGR = pd.DataFrame(GR_list)
    classificationGR.columns = ["GR_index", "GR_Rule", "GR_score", "docid", "MRCode"]
    if type == "foil":
        classificationGR.to_csv("Foil_classifier_.csv", sep='\t', index=False)
    if type == "simple":
        classificationGR.to_csv("Simple_classifier_.csv", sep='\t', index=False)


if __name__ == '__main__':
    #This allows parameters to be passed on the command line.
    testingData = sys.argv[1]
    rules = sys.argv[2]
    type = sys.argv[3]
    predict(testingData, rules, type)
