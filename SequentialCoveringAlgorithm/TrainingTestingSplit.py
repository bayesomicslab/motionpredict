import pandas as pd
import re
from sklearn.utils import shuffle
import sys
#splits data 80 20 for training and testing
def GR_DN_separation(corpus):
    print("reading csv file: new_corpus.csv ...")
    nlpCorpus = pd.read_csv(corpus, sep='\t')
    print("dropping all but GR and DN motionresultcodes ...")
    GR_DN_nlpCorpus = nlpCorpus[(nlpCorpus["MotionResultCode"] == "GR") | (nlpCorpus["MotionResultCode"] == "DN")]
    print("separating GR and DN MotionResultCodes to separate dataframes ...")
    DN_corpus = nlpCorpus[(nlpCorpus["MotionResultCode"] == "DN")]
    GR_corpus = nlpCorpus[(nlpCorpus["MotionResultCode"] == "GR")]
    print("separating GR and DN corpus to Testing and TrainingData ...")
    count_row_GR = GR_corpus.shape[0]
    count_row_DN = DN_corpus.shape[0]
    twenty_percent_GR = int(round(0.2 * count_row_GR))
    twenty_percent_DN = int(round(0.2 * count_row_DN))
    GR_training_data = GR_corpus[twenty_percent_GR:]
    GR_testing_data = GR_corpus[:twenty_percent_GR]
    DN_training_data = DN_corpus[twenty_percent_DN:]
    DN_testing_data = DN_corpus[:twenty_percent_DN]
    training_data = GR_training_data.append(DN_training_data, ignore_index=True)
    testing_data = GR_testing_data.append(DN_testing_data, ignore_index=True)
    shuffle_training_data = shuffle(training_data)
    shuffle_testing_data = shuffle(testing_data)
    print("finishing Training and Testing separation ...")
    shuffle_training_data.to_csv("TrainingData.csv", sep="\t", index=False)
    shuffle_testing_data.to_csv("TestingData.csv", sep="\t", index=False)

if __name__ == '__main__':
    corpus = sys.argv[1]
    GR_DN_separation(corpus)
