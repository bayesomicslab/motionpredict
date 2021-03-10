#packages/libraries needed to run code.
import pandas as pd
import numpy as np
import sys

'''
This script will seperate data to use for cross validation. It uses the k-fold cross validation technique where k = 5.
The function takes in the training data csv file outputed from the TrainingTestingSplit.py script. 
The outputed csv files are called
crossValTraining_1_.csv    crossValTesting_1_.csv
crossValTraining_2_.csv    crossValTesting_2_.csv
crossValTraining_3_.csv    crossValTesting_3_.csv
crossValTraining_4_.csv    crossValTesting_4_.csv
crossValTraining_5_.csv    crossValTesting_5_.csv

Command to run the script: python3 name_of_this_file.py your_training_data
'''


def crossValSplit(TrainingData):
    '''
    Reads in the training data as a dataframe and shuffles the data. Then it splits the dataframe into 5 equal parts.
    These 5 parts are the testing data. They will be outputed as csv files called crossValTesting_number_.csv, where number is the
    name for each fold.
    For each fold, it appends the rest of the folds together to create the training data for that corresponding fold.
    This then outputed as a csv file called crossValTraining_number_.csv.
    If the number on the training and testing csv files are the same, they are used together to cross validate.
    '''
    dataset = pd.read_csv(TrainingData, sep='\t')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    split = np.array_split(dataset, 5)
    for i in range(len(split)):
        trainingList = []
        for j in range(len(split)):
            if j != i:
                trainingList.append(split[j])
                merge = np.concatenate(trainingList, axis=0)

        testingValData = pd.DataFrame(split[i])
        trainingValData = pd.DataFrame(merge)
        trainingValData.columns = ["docid", "text", "MotionResultCode"]

        testingValData.to_csv("crossValTesting_" + str(i + 1) + "_.csv", sep='\t', header=True, index=False)
        trainingValData.to_csv("crossValTraining_" + str(i + 1) + "_.csv", sep='\t', header=True, index=False)

if __name__ == '__main__':
    #This allows paramters to be passed on the command line.
    TrainingData = sys.argv[1]
    crossValSplit(TrainingData)
