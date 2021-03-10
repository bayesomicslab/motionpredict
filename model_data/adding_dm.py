from scipy.stats import entropy
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import math
from multiprocessing import Pool
import random



# Convert the dictionary to be just about major codes

'''
Turns the dictIn dictionary that is being used to store all the counts of the major codes as seperate dictionarys per 
attorney into a dictionary with just of the major code values and attorney codes.

Will return newDict that is attorneyNumber:[majorCodeValueCounts]
'''
def getDictListMajor_dm(dictIn):
    newDict = {}
    pick = random.choice(list(dictIn.keys()))
    entry_form = list(dictIn[pick].keys())
    for i in dictIn:
        newEntry = [0 for j in range(len(entry_form))]

        for obj in dictIn[i]:
            idx = entry_form.index(obj)
            newEntry[idx] = dictIn[i][obj]

        newDict[i] = newEntry

    return newDict

'''
Computing the shannon index for each lawyer smoothed by a dirichlet multinomial prior

y - is the actual counts for each attorney major codes

majorVals - A list version of all the specilizations for the attorneys
majors - appending the attorney specilization to the end of its list of major codes
'''
def getEntropy_dm(y):

    majors = getDictListMajor_dm(y)

    majorVals = []
    for i in majors:
        ent_array = []
        for item in majors[i]:

            # Mean of the dirichlet multinomial with a flat prior of all 1s
            ent_array += [ (item + 1) / (np.sum(majors[i]) + len(majors[i])) ] # <--- mean dirmult


        calc_entrop = entropy(pk=ent_array, base=2)

        majorVals += [calc_entrop]
        majors[i] += [calc_entrop]

    return majorVals, majors

'''
Count the number of case major codes an attorney takes on in your data

work - the input data tsv file containing all the features. Must include CaseMajorCode and CaseAttorneyJuris to compute 
the atoorney specilization

There is no return from this function. It will output a file similar to the origial file but now with a column containing
attorney specilization
'''
def parseData(work):
    y_actual = {}

    # loading the data
    print("Working on: %s" % work)
    atnyData = pd.read_csv(work, sep="\t")
    attorneyInfo = pd.DataFrame(atnyData[extract], columns=extract)

    # Get the counts
    for index, row in attorneyInfo.iterrows():

        jurisNumber = row.CaseAttorneyJuris
        majCode = row.CaseMajorCode

        # Build dictionary of case counts
        if jurisNumber not in y_actual:
            y_actual[jurisNumber] = {}
            for j in caseTypes:
                y_actual[jurisNumber][j] = 0

            y_actual[jurisNumber][majCode] += 1

        else:

            y_actual[jurisNumber][majCode] += 1


    major, m_dict = getEntropy_dm(y_actual)

    dm_bern = []
    for i,r in atnyData.iterrows():

        if r.CaseAttorneyJuris in m_dict:
            dm_bern += [m_dict[r.CaseAttorneyJuris][-1]]
        else:
            dm_bern += [0]

    atnyData['dmEntropy'] = dm_bern

    working = work.split('.',1)
    working_f = working[0] + '_dm.' + work.split('.',1)[1]
    atnyData.to_csv(working_f,sep='\t',index = False)


if __name__ == '__main__':
    # Working Directory

    # data file that you want to use
    data_set = sys.argv[1]

    extract = ["CaseReferenceNumber", "MotionResultCode", "CaseAttorneyType", "MotionID",
               "MotionJurisNumber", "CaseAttorneyJuris", "CaseMajorCode", "CaseMinorCode",
               "MotionDocumentTypeName"]

    caseTypes = {}
    for i,r in atnyData.iterrows():
        caseTypes[r.CaseMajorCode] = 0

    parseData(data_set)









