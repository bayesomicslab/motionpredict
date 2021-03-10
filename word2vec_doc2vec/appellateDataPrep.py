from nlp_mod_pipeline import addBiandTri, tokenDocs
import nltk
import pandas as pd
import sys
import gensim
from gensim.models import Phrases
from gensim.corpora import Dictionary

#this script processes the appellate court opinions txt file and prepares it for word2vec and doc2vec
#to run: python3 appellateDataPrep.py AppellateOpinionLegalData.txt
#outputs formatted csv for word2vec and doc2vec called Appellate_Opinion_To_Be_Embedded.csv
def preparingData(data):
    nlpCorpus = pd.DataFrame()
    print("load the dataset...")
    nlpCorpus = pd.read_csv(data, sep="\t", names=["doc_count", "text"])
    len(nlpCorpus)
    print("applying functions....")
    nlpCorpus["text"] = nlpCorpus["text"].apply(tokenDocs)
    newData = nlpCorpus["text"].tolist()
    dataAndGrams = addBiandTri(newData)

    print("filtering....")
    dictionary = Dictionary(dataAndGrams)
    dictionary.filter_extremes(no_below=25, no_above=0.5)

    filtered = set([k for k, v in dictionary.token2id.items()])
    filteredDocs = [list(set(item) & filtered) for item in dataAndGrams]

    print("filter words that are less then 3 letters in length...")
    tmp = []
    for item in filteredDocs:
        t = [obj for obj in item if len(obj) > 3]
        tmp += [t]
    for i in range(len(tmp)):
        tmp[i] = " ".join(tmp[i])
    nlpCorpus.text = tmp

    print("Output the nlpCorpus....")
    nlpCorpus.to_csv("Appellate_Opinion_To_Be_Embedded.csv", sep='\t')

if __name__ == '__main__':
    data = sys.argv[1]
    preparingData(data)
