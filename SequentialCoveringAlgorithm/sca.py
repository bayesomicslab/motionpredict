#These are the packages/libraries needed to run the script.
import pandas as pd
import csv
import re
import math
import numpy as np
import sys

'''
The first parameter is your training data. The condition for generating rules is either the simple or the Foil Information Gain depending on what is chosen.
You will need to cross validate in order to find the second parameter. The cross validation split can be created using a python script in this directory.

This script outputs a csv file with 5 columns. First column is the word (rule).
Second column is the number of times the word appears in the granted documents. Third column is
the number of times the word appears in the denied documents. Fourth column is the total number of times
the word appears throughout all the documents. Fifth column is a statistic that the function chosen outputs.

The command for running this script is the following: python3 name_of_this_script.py your_data your_cutoff_score which_equation_to_use

the type argument is inputted as either "simple" or "foil"
ex) python3 sca.py TrainingData.csv 0 simple
'''
def Training_Rules(trainingData, cutoff, type):
    #This reads the training data as a csv and turns it into a pandas dataframe that is seperated by tabs.
    GR_DN_nlp_training = pd.read_csv(trainingData, sep='\t')
    print("reading training data complete...")
    #Drops all rows with NaN vlues in the text column.
    GR_DN_nlp_training.dropna(subset=["text"], inplace=True)
    '''
    Data processing:
    Creates a new dataframe that seperates each word in every document
    into a seperate row. The first column is the document id. Second column is the word. Third
    column is the motion result code. This also skips all non-english words/noise in the data.
    '''
    dfCorpusColumns = GR_DN_nlp_training.columns
    c0 = dfCorpusColumns[0]
    c1 = dfCorpusColumns[1]
    c2 = dfCorpusColumns[2]
    listDecomposedWords = []
    for index, row in GR_DN_nlp_training.iterrows():
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
    print("total words: ", len(listDecomposedWords))
    dfWords = pd.DataFrame(listDecomposedWords)
    dfWords.columns = [c0,'word',c2]

    '''
    This block of code implements the Sequential Covering Algorithm.
    '''
    Finish_run = False
    print("creating rules....")
    countRules = 0
    while not Finish_run:
        dfGRwords = dfWords[dfWords[c2] == 'GR']
        if dfGRwords.shape[0] == 0:
            break
        GRgrouped = dfGRwords.groupby(['word', c2])[['word']].count()
        GRgrouped = GRgrouped.rename(columns={'word': 'count'})
        GRgrouped = GRgrouped.reset_index()
        print("GR row count ", GRgrouped.shape[0])

        dfDNwords = dfWords[dfWords[c2] == 'DN']
        DNgrouped = dfDNwords.groupby(['word', c2])[['word']].count()
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
        np.seterr(divide = 'ignore')
        #foil or simple
        if type == "foil":
            numOfGRDocs = dfWords[dfWords.MotionResultCode == 'GR'].shape[0]
            totalDocCount = dfWords.shape[0]
            constant = (numOfGRDocs / totalDocCount)
            GRstat = (mergedFrames.GR_count / mergedFrames.Total_Count)
            #FOIL Info-Gain Function:
            mergedFrames['function'] = (mergedFrames.GR_count)*(np.log2(GRstat) - math.log2(constant))
        if type == "simple":
            #Simple function
            mergedFrames['function'] = (mergedFrames.GR_count + 1)/(mergedFrames.Total_Count + 2)
        #Finds highest valued word from the function.
        mvalue = mergedFrames[mergedFrames['function'] == mergedFrames['function'].max()]
        #breaks loop when the highest valued word using the foil function gets to the cutoff parameter.
        for j, row in mvalue.iterrows():
            if mvalue.loc[j, 'function'] < float(cutoff):
                Finish_run = True
                break
        if Finish_run:
            break
        #writes rule to csv file.
        if type == "foil":
            if countRules == 0:
                mvalue.to_csv("FOIL_Rules_" + str(cutoff) + ".csv", sep='\t', header=True, index=False)
            else:
                mvalue.to_csv("FOIL_Rules_" + str(cutoff) + ".csv", sep='\t', mode='a', header=False, index=False)
        else:
            if countRules == 0:
                mvalue.to_csv("Simple_Rules_" + str(cutoff) + ".csv", sep='\t', header=True, index=False)
            else:
                mvalue.to_csv("Simple_Rules_" + str(cutoff) + ".csv", sep='\t', mode='a', header=False, index=False)

        #Dropping data section
        mvalue = mvalue.reset_index()
        docid_to_be_removed = []
        #appends all documents that needs to be removed from the training data to a list.
        for i, r in mvalue.iterrows():
            if len(dfWords) == 0:
                break
            for j, rw in dfWords.iterrows():
                if dfWords.loc[j, 'word'] == mvalue.loc[i, 'word']:
                    docid_to_be_removed.append(dfWords.loc[j, 'docid'])
                if len(dfWords) == 0:
                    break
        #skips all words from the documents that need to be removed list, and appends all remaining documents to another list.
        unique_docid = list(set(docid_to_be_removed))
        print("creating remaining words list ...")
        remaining_words = []
        for i, rw in dfWords.iterrows():
            if rw[0] in unique_docid:
                continue
            remaining_words.append(rw)

        #creates dataframe from the remaining documents and allows to loop back to recalculate and find a new rule.
        dfRemainingWords = pd.DataFrame(remaining_words)
        dfRemainingWords.columns = [c0,'word',c2]

        dfWords = dfRemainingWords

        countRules += 1

        print("total dfWords left, count Rule .... ", countRules, len(dfWords))

        if len(dfWords) == 0:
            Finish_run = True
    print("Rules generated...")

if __name__ =='__main__':
    #This allows parameters to be passed in the command line.
    trainingData = sys.argv[1]
    cutoff = sys.argv[2]
    type = sys.argv[3]
    Training_Rules(trainingData, cutoff, type)
