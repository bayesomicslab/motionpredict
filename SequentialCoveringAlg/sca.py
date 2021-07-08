#These are the packages/libraries needed to run the script.
import pandas as pd
import csv
import re
import math
import numpy as np
import sys
import os
import threading
import statistics
from datetime import datetime
from sklearn.metrics import accuracy_score
from DataPrep.train_test_split import *
'''
Trains, validates, and tests the sequential covering algorithm on simple and foil info gain criteria.
Outputs rules_criteria.txt and predictions_criteria.txt depending on the criteria.
'''

class SequentialCoveringAlg:
    def __init__(self, data, cutoff, type, output=None):
        self.data = data
        self.cutoff = cutoff
        self.type = type
        self.output = output
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    #cleans and helps for the sequential covering alg.
    def seperateTextToWords(self, remainingData):
        #Drops all rows with NaN vlues in the text column.
        remainingData.dropna(subset=["text"], inplace=True)
        #Cleans data and creates dataframe with docid, word, MotionResultCode
        dfCorpusColumns = remainingData.columns
        c0 = dfCorpusColumns[0]
        c1 = dfCorpusColumns[1]
        c2 = dfCorpusColumns[2]
        listDecomposedWords = []
        for index, row in remainingData.iterrows():
            idCase = row[c0]
            tvCode = row[c2]
            listTX = row[c1]
            listTX = listTX.lstrip()
            listTX = listTX.rstrip()
            lWords = listTX.split()
            for i in range(len(lWords)):
                try:
                    lWords[i].encode(encoding='utf-8').decode('ascii')
                except UnicodeDecodeError:
                    continue
                unnecessary_words = re.findall(r'((\w)\2{2,})', lWords[i])
                if len(unnecessary_words) > 0:
                    continue
                listRowItems = []
                listRowItems.append(idCase)
                listRowItems.append(lWords[i])
                listRowItems.append(tvCode)
                listDecomposedWords.append(listRowItems)

        dfWords = pd.DataFrame(listDecomposedWords)
        dfWords.columns = [c0,'word',c2]
        return dfWords

    '''
    trains to develop rules.
    '''
    def Training_Rules(self):
        #This reads the training data as a csv and turns it into a pandas dataframe that is seperated by tabs.
        GR_DN_nlp_training = pd.read_csv(self.data, sep='\t')
        dfWords = self.seperateTextToWords(GR_DN_nlp_training)
        print("creating rules....")
        rule_list = []
        #----------------------------SEQUENTIAL COVERING START----------------------------
        #INITIAL RULE FOR FOIL: if a word is in a GR document,  it is considered a rule
        numGR = GR_DN_nlp_training[GR_DN_nlp_training["MotionResultCode"] == 1].shape[0]
        totaldocs = GR_DN_nlp_training.shape[0]
        constant = numGR / totaldocs
        #for cross validation
        highest_thresholds = []
        print(f"applying sca for {self.cutoff}...")
        while True:
            dfGRwords = dfWords[dfWords["MotionResultCode"] == 1]
            dfDNwords = dfWords[dfWords["MotionResultCode"] == 0]

            if dfGRwords.shape[0] == 0:
                break

            GRgrouped = dfGRwords.groupby(['word', "MotionResultCode"])[['word']].count()
            GRgrouped = GRgrouped.rename(columns={'word': 'count'})
            GRgrouped = GRgrouped.reset_index()

            DNgrouped = dfDNwords.groupby(['word', "MotionResultCode"])[['word']].count()
            DNgrouped = DNgrouped.rename(columns={'word': 'count'})
            DNgrouped = DNgrouped.reset_index()

            #----------merge 2 dataframes--------------------
            mergedFrames = pd.merge(GRgrouped, DNgrouped, on=['word'], how='outer')
            mergedFrames.drop(['MotionResultCode_x', 'MotionResultCode_y'], axis=1, inplace=True)
            mergedFrames = mergedFrames.rename(columns={'count_x': 'GR_count', 'count_y': 'DN_count'})
            mergedFrames.fillna(0, inplace=True)
            mergedFrames['GR_count'] = mergedFrames['GR_count'].astype(float)
            mergedFrames['DN_count'] = mergedFrames['DN_count'].astype(float)
            mergedFrames['Total_Count'] = mergedFrames["GR_count"] + mergedFrames["DN_count"]
            mergedFrames.drop(index=mergedFrames[mergedFrames["GR_count"] == float(0)].index, inplace=True)
            mergedFrames.reset_index(drop=True, inplace=True)
            #--------------------------------CRITERIA----------------------------------------------
            if self.type == "simple":
                #simple
                mergedFrames['function'] = (mergedFrames.GR_count + 1) / (mergedFrames.Total_Count + 2)
            else:
                #foil info gain
                GRstat = (mergedFrames["GR_count"] / mergedFrames["Total_Count"])
                mergedFrames['function'] = (mergedFrames["GR_count"])*(np.log2(GRstat) - math.log2(constant))

            #--------------------------------PREP DATA REMOVAL AND ADDING RULE----------------
            #sort
            mergedFrames = mergedFrames.sort_values("function", ascending=False)
            highest_valued_words = mergedFrames[mergedFrames["function"] == mergedFrames["function"].max()]
            #for cross validation
            highest_thresholds.extend(list(set(highest_valued_words["function"].tolist())))
            # check if value of rule reaches threshold
            if all(word_freq <= float(self.cutoff) for word_freq in highest_valued_words["function"].tolist()):
                break
            #check if there are no granted words or if number of granted is equal to number of denied
            rules_to_add_and_remove = highest_valued_words["word"].tolist()
            rule_list.extend(rules_to_add_and_remove)

            #-------------------------------REMOVE DATA BASED ON RULE-----------------------
            # for rule in rules_to_add_and_remove:
            #     GR_DN_nlp_training.drop(index=GR_DN_nlp_training[GR_DN_nlp_training["text"].str.contains(rule)].index, inplace=True)
            GR_DN_nlp_training = GR_DN_nlp_training[~GR_DN_nlp_training["text"].str.contains('|'.join(rules_to_add_and_remove))]

            df_docid_to_remove = dfWords[dfWords["word"].str.contains('|'.join(rules_to_add_and_remove))]
            docid_to_be_removed = df_docid_to_remove["docid"].tolist()
            dfWords = dfWords[~dfWords["docid"].isin(docid_to_be_removed)]
            #remove words from word dataframe
            GR_DN_nlp_training.reset_index(drop=True, inplace=True)
            #will print how mcuh data is left after removal. Ucomment if needed
            # print("Length of overall data: ", GR_DN_nlp_training.shape[0])
            # print("Length of dfWords: ", dfWords.shape[0])
            if GR_DN_nlp_training.shape[0] == 0:
                break

        #-----------------RETURN RULE LIST--------------------
        #output rules to txt
        if self.output == "yes":
            print("writing rules to output txt file....")
            cwd = os.getcwd()
            dir = f"Rules_{self.cutoff}_{self.type}"
            path = os.path.join(cwd, dir)
            try:
                os.mkdir(path)
                print("Directory '% s' created" % dir)
            except OSError as err:
                print("Directory '% s' already exists" % dir)

            with open(os.path.join(path, f"rules_{self.type}_{self.cutoff}_.txt"), "w") as f:
                for r in rule_list:
                    f.write(r + "\n")
        #write highest thresholds in output txt
        # with open("highest_thresholds_foil.txt", "w") as f1:
        #     highest_thresholds = sorted(list(set(highest_thresholds)))
        #     for thshhld in highest_thresholds:
        #         f1.write(str(thshhld) + "\n")
        return rule_list
        print("Done.")

    #-------------------------------------------------------------------------------------------------------------------------
    '''
    predicts testing data.
    '''
    def predict(self, rules, testing_data):
        print("reading in csv files ...")
        predictions = pd.read_csv(testing_data, sep='\t')
        print("classifying GR....")
        predictions["MotionResultCode"].values[:] = 0
        for rule in rules:
            predictions.loc[predictions["text"].str.contains(rule), "MotionResultCode"] = 1
        #create output for predicctions. Uncomment if needed
        # cwd = os.getcwd()
        # path = os.path.join(cwd, output_folder_name)
        # try:
        #     os.mkdir(path)
        #     print("Directory '% s' created" % output_folder_name)
        # except OSError as err:
        #     print("Directory '% s' already exists" % output_folder_name)
        # predictions.to_csv(os.path.join(path, f"predictions_{self.type}_{self.cutoff}_.csv"), sep="\t")
        prediction_labels = predictions["MotionResultCode"].tolist()
        return prediction_labels

    '''
    computes classification accuracy.
    '''
    def classification_accuracy(self, predictions, testingData):
        #read testing data
        print("getting classification accuracy...")
        testing_data = pd.read_csv(testingData, sep='\t')
        test_labels = testing_data["MotionResultCode"].tolist()
        # predictions_labels = predictions["MotionResultCode"].tolist()
        ca = accuracy_score(test_labels, predictions)
        print(ca)
        return ca


#--------------------------------------------------------------------------------------------------------------


def crossValidation(criteria_type, count):
    if criteria_type == "simple":
        list_of_threshold = np.arange(0.5, 0.9, 0.01)
    else:
        list_of_threshold = np.arange(0, 50, 1)

    output_folder_name = f"CV_Results_Fold_{count}_{criteria_type}"
    cwd = os.getcwd()
    path = os.path.join(cwd, output_folder_name)
    os.mkdir(path)


    for threshold in list_of_threshold:
        #output for results
        training_data = f"./DataPrep/CrossValidationData/CV_TrainData_{count}.csv"
        testing_data = f"./DataPrep/CrossValidationData/CV_TestData_{count}.csv"
        c = SequentialCoveringAlg(training_data, threshold, criteria_type)
        rules = c.Training_Rules()
        predictions = c.predict(rules, testing_data)
        ca = c.classification_accuracy(predictions, testing_data)
        print(str(threshold) + " : " + str(ca) + "\n")

        results = str(ca) + "\n"
        with open(os.path.join(path, f"results.txt"), "a") as f:
            f.write(results)

def startMultithreadCrossValidation(criteria_type):
    start_time = datetime.now()
    threads = []
    for i in range(5):
        t = threading.Thread(target=crossValidation, args=[criteria_type, i])
        threads.append(t)
        t.start()

    [thread.join() for thread in threads]

    end_time = datetime.now()
    print("--------------------Exit time: ", (end_time - start_time).total_seconds(), " ----------------------------")


    
def findThreshHold(criteria, start, step):
    directories = []
    for i in range(5):
        directories.append(os.path.join(os.getcwd(), f"CV_Results_Fold_{i}_{criteria}"))

    list_of_values = {}
    for dir in directories:
        for filename in os.listdir(dir):
            list_num = []
            count = start
            with open(os.path.join(dir, filename)) as f:
                for line in f:
                    line = line.rstrip("\n")
                    num = float(line)
                    if count in list_of_values:
                        list_of_values[count].extend([num])
                    else:
                        list_of_values[count] = [num]
                    count += step

    average_d = {}
    for key, value in list_of_values.items():
        average_d[key] = statistics.mean(value)
    print("best threshold value to stop making rules: ", max(average_d, key=average_d.get))

#---------------------------------------------

if __name__ =='__main__':

    #This allows parameters to be passed in the command line.
    #cross validation:
    if sys.argv[1] == "splitdata":
        train_test_main()

    elif sys.argv[1] == "CV":
        if sys.argv[2] == "simple":
            startMultithreadCrossValidation("simple")
            findThreshHold("simple", 0.5, 0.01)
        else:
            startMultithreadCrossValidation("foil")
            findThreshHold("foil", 1, 1)

    else:
        training_data = sys.argv[1]
        threshold = sys.argv[2]
        criteria_type = sys.argv[3]
        c = SequentialCoveringAlg(training_data, threshold, criteria_type, "yes")
        rules = c.Training_Rules()
        predictions = c.predict(rules, "./DataPrep/TrainTest/TestingData.csv")
        c.classification_accuracy(predictions, "./DataPrep/TrainTest/TestingData.csv")

#run split data: python3 sca.py splitdata
#run cross validation: python3 sca.py CV simple
#run final train, test: python3 sca.py ./DataPrep/TrainTest/TrainingData.csv 0.75 simple
