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
'''
Adding the bigrams and trigrams to the capture any potential phrases that
might be important in the data
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
