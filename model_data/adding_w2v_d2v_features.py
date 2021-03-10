import pandas as pd
import sys
import numpy as np
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim.models import Phrases
from gensim.corpora import Dictionary
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from os import listdir
import pickle



'''
Adding the bigrams and trigrams to the capture any potential phrases that
might be important in the data
 
docs - text corpus
return the text corpus with trigrams and bigrams added to it
'''
def addBiandTri(docs):
    # Add bigrams and trigrams to docs (only ones that appear 5 times or more).
    bigram = Phrases(docs, min_count=5)
    trigram = Phrases(bigram[docs])

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
           	    docs[idx].append(token)
    return docs

'''
Some processing on the text files. we are going through and doing some of the
basic NLP steps. We start by tokenizing the words, then removing  stop words,
lemmatize the words, and stem the words.

inTxt - the text corpus 

:return - the tokenized version of the input
'''
def tokenDocs(inTxt):

    # Intialize the stop word calculator, lemmatizer, and stemmer
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    snowball_stemmer = nltk.stem.SnowballStemmer('english')

    # Tokenize the documents
    tokens = nltk.word_tokenize(inTxt)

    # Convert all things to lower case, make sure things are numeric, remove stopwords,
    # lemmatize and stem things
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in nltk_stopwords]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    tokens = [snowball_stemmer.stem(token) for token in tokens]

    return tokens

'''
Mapping the CaseReferenceNumber and documentID so that we can properly link the sparse data and the dense data

word_embeding -  All of the sparse data 
map - the docID and CaseReferenceNumber matching dataFrame

:return a dataframe with a column that adds a column with a case reference number
'''
def get_case_refNum(word_embeding, map):

    refNum_list = []
    for i, r in word_embeding.iterrows():
        id = int(r.docid)
        refNum = map[map.docid == id].CaseReferenceNumber.tolist()[0]
        refNum_list += [refNum]
    word_embeding['caseRef'] = refNum_list

    return word_embeding

'''
Mathcing the database features with the complain documenets through the case reference number and document id number.
Here we are adding the features of the sparse data to the dense data each word is added as a new column to the data 

new_features - the sparse data 
dense - data_base features

:return new_data_out - the new combined dataset
'''
def get_newData_Sets(new_features, dense):

    motions2 = pd.read_csv(dense, compression='gzip')

    m_cols = motions2.columns
    if 'CaseReferenceNumber' not in m_cols:
        motions2 = pd.read_csv(dense, sep='\t', compression='gzip')

    new_data_out = pd.DataFrame(columns=motions2.columns)
    new_data_out[new_data_out.columns] = motions2[motions2.columns]

    cols = new_features.columns[:-2]
    for item in cols:
        new_data_out[item] = np.nan

    done = []
    for i,r in new_features.iterrows():

        if i in done:
            continue

        idxs = new_data_out[new_data_out.CaseReferenceNumber == r.caseRef].index.tolist()

        for item in idxs:

            if item in done:
                continue
            else:

                for j in range(len(cols)):

                    new_data_out.at[item, cols[j]] = list(r[list(cols)])[j]

    return new_data_out


'''
This function will remove any words in the corpus that are not in the rules.

ruleDF - a dataframe with all the generated rules in it
corpusData - dataFrame with all of the complaint documents in it

:return newDF_holder which has only the words from the generated rules
'''
def remove_wrords(ruleDF,corpusData):

    newDF_holder = pd.DataFrame(columns=corpusData.columns)
    new_corpus_data = []
    rules = list(ruleDF.word)

    # make rule list
    rules = [w.strip() for w in rules]

    for i, r in corpusData.iterrows():
        current_text = r.text
        data = []

        # keeping words that are only in the rules
        for word in current_text:
            if word.strip() in rules:
                data += [word]

        new_corpus_data += [data]
    newDF_holder.text = new_corpus_data
    newDF_holder.docid = list(corpusData.docid)

    return newDF_holder

if __name__ == '__main__':

    # documents = 'data/original/strikemotion_code_T_V_complaint_doc_ocr.txt.gz'
    documents = sys.argv[1]
    # denseData = 'data/new_test/motionStrike_TVcodes_data.tsv.gz'
    denseData = sys.argv[2]
    # trnslationTable = 'data/original/judcaseid_docid_translationtable.tsv.gz'
    trnslationTable = sys.argv[3]
    # simple = 'data/simple_Rules.csv'
    # foil = 'data/foil_Rules.csv'
    simple = sys.argv[4]
    foil = sys.argv[5]
    # word2vec = 'word2vec/'
    # doc2vec = 'doc2vec/'
    word2vec = sys.argv[6]
    doc2vec = sys.argv[7]
    bbe_file = sys.argv[8]


    nlpCorpus = pd.read_csv(documents, sep='\t', compression='gzip', header=None,
                            names=["page", "docid", "text", "certainty"])
    # We do not need the certainty column and we want to drop rows that are
    # are missing data
    nlpCorpus = nlpCorpus.drop(columns=['certainty'])
    nlpCorpus = nlpCorpus.dropna()
    nlpCorpus = nlpCorpus.groupby('docid')['text'].apply(' '.join).reset_index()


    # Apply the functions above to the data in a way that it will yield all
    # the proccessed info
    nlpCorpus['text'] = nlpCorpus['text'].apply(tokenDocs)
    newData = nlpCorpus['text'].tolist()

    # Adding bigram and trigrams accordingly
    dataAndGrams = addBiandTri(newData)

    # Now make a gensim dictionary to filter out specific words that do not
    # show up in 2 or more documents and words that appear in more then 50
    # percent of the documents
    dictionary = Dictionary(dataAndGrams)
    dictionary.filter_extremes(no_below=25, no_above=0.5)

    # Update the modified version of the filtered words
    filtered = set([k for k, v in dictionary.token2id.items()])
    filteredDocs = [list(set(item) & filtered) for item in dataAndGrams]

    # Now fileter words that are less then 3 letters in length
    tmp = []
    for item in filteredDocs:
        t = [obj for obj in item if len(obj) > 3]
        tmp += [t]

    nlpCorpus.text = tmp

    mapping = pd.read_csv(trnslationTable, sep='\t', compression='gzip')
    mapping.columns = ['docid', 'CaseReferenceNumber']
    motions = pd.read_csv(denseData, sep='\t', compression='gzip')

    extract = ["CaseReferenceNumber", "MotionResultCode"]
    motions = pd.DataFrame(motions[extract], columns=extract)

    nlpCorpus["MotionResultCode"] = " "
    for i, r in nlpCorpus.iterrows():
        id = int(r.docid)
        refNum = mapping[mapping.docid == id].CaseReferenceNumber.tolist()[0]
        result = motions[motions.CaseReferenceNumber == refNum].MotionResultCode.tolist()[0]
        r.MotionResultCode = result

    # getting the new features
    simple_Rules = pd.read_csv(simple, sep='\t')
    foil_Rules = pd.read_csv(foil, sep='\t')

    word2vecfiles = listdir(word2vec)
    doc2vecfiles = listdir(doc2vec)

    # developing the new datasets with the doc2vec features
    for item in doc2vecfiles:

        infile = open(doc2vec + item, 'rb')
        new_dict = pickle.load(infile)

        # make a dataframe that contains all of the doc2vec embeddings
        x = np.zeros([len(new_dict), 300],dtype=np.float64)
        id = []
        keys = list(new_dict.keys())
        for k in range(len(keys)):
            id += [keys[k]]
            x[k][:] = new_dict[keys[k]][:]

        cols =  [i for i in range(300)]
        frame = pd.DataFrame(data=x,columns =cols)
        frame['docid'] = id

        # output the data and connecting sparse data and dense data
        refs = get_case_refNum(frame, mapping)
        new_data = get_newData_Sets(refs, denseData)

        data_from = item.split('/')[-1].split('.',1)[0]
        dd = denseData.split('/')[-1].split('.',1)[0]
        new_name = denseData.split('.',1)[0] + '_' + data_from + '.' + denseData.split('.',1)[-1]
        new_data.to_csv(new_name, sep='\t',compression='gzip',index=False)

        data_from = item.split('/')[-1].split('.', 1)[0]
        dd = bbe_file.split('/')[-1].split('.', 1)[0]
        new_name = bbe_file.split('.', 1)[0] + '_' + data_from + '.' + bbe_file.split('.', 1)[-1]

        bbe_data_foilTFIDF = get_newData_Sets(refs, bbe_file)
        bbe_data_foilTFIDF.to_csv(new_name, sep='\t',compression='gzip', index=False)



    # Developing the TFIDF-rules Version

    # get the inverse document frequency weights for the simple word rules
    simple_Remove_tfidf = remove_wrords(simple_Rules, nlpCorpus)
    vectorizer = TfidfVectorizer()
    textInfo = list(simple_Remove_tfidf.text)
    textInfo = [' '.join(item) for item in textInfo]
    X = vectorizer.fit(textInfo)

    simp_dict = dict(
        [(w, X.idf_[i]) for w, i in X.vocabulary_.items()])

    # get the inverse document frequency weights for the foil word rules
    foil_Remove_tfidf = remove_wrords(foil_Rules, nlpCorpus)
    vectorizer = TfidfVectorizer()
    textInfo = list(foil_Remove_tfidf.text)
    textInfo = [' '.join(item) for item in textInfo]
    X = vectorizer.fit(textInfo)

    foil_dict = dict(
        [(w, X.idf_[i]) for w, i in X.vocabulary_.items()])

    for item in word2vecfiles:

        infile = open(word2vec + item, 'rb')
        new_dict = pickle.load(infile)

        new_cols = ['f%s' % xx for xx in range(300)]

        data_simp = pd.DataFrame(columns=new_cols)
        data_foil = pd.DataFrame(columns=new_cols)
        k = 0

        for i, r in nlpCorpus.iterrows():
            x = np.zeros(shape=(len(simple_Rules), 300), dtype=np.float64)
            y = np.zeros(shape=(len(foil_Rules), 300), dtype=np.float64)

            # Find if the rule was in the documents and make a matrix of it and multiple the idf onto that vector
            for j in range(len(simple_Rules.word.tolist())):

                if simple_Rules.word.tolist()[j] in r.text and simple_Rules.word.tolist()[j] in new_dict:
                    x[j, :] = new_dict[simple_Rules.word.tolist()[j]] * simp_dict[simple_Rules.word.tolist()[j]]

            for j in range(len(foil_Rules.word.tolist())):
                if foil_Rules.word.tolist()[j] in r.text and foil_Rules.word.tolist()[j] in new_dict:
                    y[j, :] = new_dict[foil_Rules.word.tolist()[j]] * foil_dict[foil_Rules.word.tolist()[j]]

            # take the mean of the vectors as the features
            word_vec_simp = np.mean(x, axis=0)
            word_vec_foil = np.mean(y, axis=0)

            data_simp.loc[k] = list(word_vec_simp)
            data_foil.loc[k] = list(word_vec_foil)
            k += 1

        data_simp['docid'] = nlpCorpus.docid
        data_foil['docid'] = nlpCorpus.docid

        # connecting sparse data and dense data
        refs_simp = get_case_refNum(data_simp, mapping)
        new_data_simp = get_newData_Sets(refs_simp, denseData)
        refs_foil = get_case_refNum(data_foil, mapping)
        new_data_foil = get_newData_Sets(refs_foil, denseData)

        # output data
        data_from = item.split('/')[-1].split('.', 1)[0]
        dd = denseData.split('/')[-1].split('.', 1)[0]
        new_name_simp = denseData.split('.', 1)[0] + '_simp_' + data_from + '.' + denseData.split('.', 1)[-1]
        new_name_foil = denseData.split('.', 1)[0] + '_foil_' + data_from + '.' + denseData.split('.', 1)[-1]
        new_data_simp.to_csv(new_name_simp, sep='\t',compression='gzip', index=False)
        new_data_foil.to_csv(new_name_foil, sep='\t',compression='gzip', index=False)

        # connecting sparse data and dense data
        bbe_data_simp = get_newData_Sets(refs_simp, bbe_file)
        bbe_data_foil = get_newData_Sets(refs_foil, bbe_file)

        data_from = item.split('/')[-1].split('.', 1)[0]
        dd = bbe_file.split('/')[-1].split('.', 1)[0]
        new_name_simp = bbe_file.split('.', 1)[0] + '_simp_' + data_from + '.' + bbe_file.split('.', 1)[-1]
        new_name_foil = bbe_file.split('.', 1)[0] + '_foil_' + data_from + '.' + bbe_file.split('.', 1)[-1]

        bbe_data_simp.to_csv(new_name_simp, sep='\t',compression='gzip', index=False)
        bbe_data_foil.to_csv(new_name_foil, sep='\t', compression='gzip', index=False)
