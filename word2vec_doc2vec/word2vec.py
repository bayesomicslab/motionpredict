import pandas as pd
import gensim
import re
import numpy as np
import pickle
import sys

''' Applies pretrained word2vec on complain documents or appellate court opinions and returns a pickled dictionary
    with the word as the key and the vector as the value
    
    to run: python3 word2vec.py your_data pretrained_model
    
'''
def word2vec(data, model):
    m = gensim.models.Word2Vec.load(model)

    #data processing. seperate into seperate words
    df = pd.read_csv(data, sep="\t")
    text = df["text"]
    text = text.dropna()
    textList = text.values.tolist()

    sentenceList = []
    for word in textList:
        sentenceList.append(word.split(" "))

    print("creating embeddings...")
    embeddings = {}
    for sentence in sentenceList:
        for w in sentence:
            try:
                embeddings[w] = m.wv[w]
            except KeyError:
                continue

    name = input("Enter the name of the model (no spaces): ")
    print("Pickling...")
    to_pickle = open(f"word2vec_on_{data[:-4]}_{name}.pkl", "wb")
    pickle.dump(embeddings, to_pickle)
    to_pickle.close()


if __name__== '__main__':
    data = sys.argv[1]
    model = sys.argv[2]
    word2vec(data, model)
