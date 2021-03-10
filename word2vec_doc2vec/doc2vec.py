#packages/libraries needed to run code.
from gensim.models import Doc2Vec
import pandas as pd
import sys

'''
Embedding the appellate data or the complaint data using pretrained doc2vec
First parameter is the data that needs to be embedded
The second parameter is the file that contains a pretrained doc2vec model.
This script turns the documents into lower dimensional vectors by applying a pretraiend model to it.
The output for this script is a pickled dataframe stored in a csv file called doc2vec.csv
that contains 3 columns: The docid, the embedded text as lower dimensional vectors
(size depends on the pretrained model), and the MotionResultCode.

Command to run script: python3 name_of_this_script your_data your_pretrained_model
    ie. python3 doc2vec.py ./Appellate_Opinion_To_Be_Embedded.csv ./enwiki_dbow/doc2vec.bin
'''

def doc2vec(data, file_model):
    #reads csv file as dataframe and drops all rows with NaN values.
    df = pd.read_csv(data, sep='\t')

    df.dropna(inplace=True)
    #loads the pretrained model using gensim's Doc2Vec module.
    model = Doc2Vec.load(file_model)
    '''
    loops through the corpus dataframe and turns the dataframe to a list of list with the
    text column as a list of words.
    '''
    row_list = []
    for i, r in df.iterrows():
        if data == "./Appellate_Opinion_To_Be_Embedded.csv":
            my_list = [r.doc_count, list(r.text.split(" "))]
            row_list.append(my_list)

        else:
            my_list = [r.docid, list(r.text.split(" ")), r.MotionResultCode]
            row_list.append(my_list)

    #uses pretrained model to turn the list of words(documents) to vectors using a gensim module.
    docVecList = []
    for i in row_list:
        if data == "./Appellate_Opinion_To_Be_Embedded.csv":

            vecItem = [i[0], model.infer_vector(i[1])]
            docVecList.append(vecItem)
        else:
            vecItem = [i[0], model.infer_vector(i[1]), i[2]]
            docVecList.append(vecItem)

    for i in range(len(docVecList)):
        print(docVecList[i])
    #converts list of vectorized docs to a dataframe and pickles it.
    if data == "./Appellate_Opinion_To_Be_Embedded.csv":
        doc2Vecdf = pd.DataFrame(docVecList, columns=["doc_count", "text"])
        doc2Vecdf.to_pickle("doc2vec_for_appellate.pkl")

    else:
        doc2Vecdf = pd.DataFrame(docVecList, columns=["docid", "text", "MotionResultCode"])
        doc2Vecdf.to_pickle("doc2vec_for_complaint.pkl")



if __name__ == "__main__":
    #allows paramters to be set in command line
    corpus = sys.argv[1]
    fmodel = sys.argv[2]
    doc2vec(corpus, fmodel)
