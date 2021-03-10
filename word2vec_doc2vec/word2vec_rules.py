import pandas as pd
import gensim
import re
import numpy as np
import pickle
import sys

''' Applies pretrained word2vec on rules and returns a pickled dictionary
    with the rule as the key and the vector as the value
    
    to run: python3 word2vec_rules.py your_data pretrained_model
    ie) python3 word2vec_rules.py Rules_FOIL_.csv apnews_sg/word2vec.bin
'''
def word2vec(rules, model):
    m = gensim.models.Word2Vec.load(model)

    #data processing. seperate into seperate words
    rules_df = pd.read_csv(rules, sep="\t")
    print("turning rules into a list...")
    rules_df.dropna(subset=["word"], inplace=True)
    listOfRules = []
    for i, r in rules_df.iterrows():
        listOfRules.append(r.word.strip(" "))
    print("using model on rules to create embeddings...")

    ruleEmbeddings = {}
    for rule in listOfRules:
        try:
            ruleEmbeddings[rule] = m.wv[rule]
        except KeyError:
            continue


    name = input("Enter the name of the model (no spaces): ")
    print("Pickling...")
    to_pickle = open(f"word2vec_on_rules_using_{name}.pkl", "wb")
    pickle.dump(ruleEmbeddings, to_pickle)
    to_pickle.close()


if __name__== '__main__':
    rules = sys.argv[1]
    model = sys.argv[2]
    word2vec(rules, model)
