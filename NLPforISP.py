import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import unicodedata
import re

import nltk
from nltk.corpus import stopwords

import gensim

'''
useful tutorials:
- n-grams using NLTK : https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460
- a good example notebook : https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb
- NLTK and gensim tutorial: https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
- NLTK and gensim tutorial : https://towardsdatascience.com/introduction-to-nlp-part-5b-unsupervised-topic-model-in-python-ab04c186f295
'''

def preprocess(text, additional_stopwords = [''], wlen = 4, stem = True):
	'''
	A simple function to clean up the data. All the words that
	are not designated as a stop word is then lemmatized and (optionally) stemmed
	after encoding and basic regex parsing are performed.
	
	originally from here : https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460
	with modifications by AMG

	arguments
	text : (string, required) a string of words (not a list)
	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming
	'''
	
	# define the lemmatizer, stemmer and stopwords
	wnl = nltk.stem.WordNetLemmatizer()
	stemmer = nltk.stem.SnowballStemmer('english')
	#stemmer = nltk.stem.PorterStemmer()
	stopwords = nltk.corpus.stopwords.words('english') + additional_stopwords
	
	# initial simple regex parsing and create a list of words
	text = (unicodedata.normalize('NFKD', text)
		.encode('ascii', 'ignore')
		.decode('utf-8', 'ignore')
		.lower())
	words = re.sub(r'[^\w\s]', '', text).split()
	
	# pass through the lemmatizer and (optionally) the stemmer
	processed = []
	for word in words:
		if (word not in stopwords and len(word) >= wlen):
			w = wnl.lemmatize(word)
			#print(word, wnl.lemmatize(word), stemmer.stem(word), stemmer.stem(w))
			if (stem):# and not w.endswith('e')):
				processed.append(stemmer.stem(w))
			else:
				processed.append(w)
	
	return processed

def getStringOfWords(df, column_number):
	'''
	returns a string of words from a dataframe that contains a column filled with rows of answers for a given question

	arguments
	df : (pandas DataFrame, required) data set containing rows of responses to one (or more) question(s)
	column_number : (integer, required) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	'''

	# convert the answers column to a list
	list_of_answers = df[df.columns[column_number]].tolist() 

	# convert that list to a long string
	string_of_answers = ''.join(str(list_of_answers))

	return string_of_answers


def getNgrams(text, n, additional_stopwords = [''], wlen = 3, stem = True, returnArray = False):
	'''
	returns ngrams for a given string of words as pandas Series (by default) of the form
	index                                    value
	(bigram0_word1, ... bigram0_wordn)       bigram0_n_ocurrences
	(bigram1_word1, ... bigram1_wordn)       bigram1_n_ocurrences
	...


	If returnArray == True, then the code returns a 2d numpy array of the form:
	[ [(bigram0_word1, ... bigram0_wordn), bigram0_n_ocurrences],
	  [(bigram1_word1, ... bigram1_wordn), bigram1_n_ocurrences], 
	  ...
	]

	arguments
	text : (string, required) a string of words (not a list)
	n : (integer, required) the number of words to select (e.g., n=2 returns bigrams)
	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	------

	useful tutorial on n-grams using NLTK : https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460

	'''

	processed_words = preprocess(text, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)
	ngramsSeries = pd.Series(nltk.ngrams(processed_words, n)).value_counts().sort_values(ascending = False)
	ngramsArray = np.array(ngramsSeries.reset_index().values.tolist(), dtype = "object")

	if (returnArray):
		return ngramsArray
	else:
		return ngramsSeries

def plotNgrams(ngrams, N, color = 'gray', ax = None):
	'''
	given an ngram in the format of a pandas Series (as outputted by default from getNgrams), this code generates
	a horizontal bar plot of the N most frequency ngrams.  The code returns the plot and axis objects if the ax arguement is None; 
	if the user supplies an ax argument, nothing is returned.

	arguments
	ngram : (pandas Series, required) contains the ngram data, as output from getNgrams
	N : (integer, required) the number of entries (rows in the ngram Series) to show in the figure
	color : (string, optional) the color of the bars
	ax : (pyplot axis object, optional) the axis object where the figure should be placed (of None, figure and axis objects are created)
	'''

	if (ax is None):
		f, ax = plt.subplots(1, 1, figsize = (5, 8))

	ind = np.arange(N)

	ngrams_plot = ngrams[0:N].sort_values()
	ax.barh(ind, ngrams_plot, 0.9, color = color)
	ax.set_yticks(ind)
	_ = ax.set_yticklabels(ngrams_plot.index.str.join(sep=' '))
	_ = ax.set_title(str(N) + ' Most Frequently Occuring N-grams')
	_ = ax.set_xlabel('# of Occurances')

	if (ax is None):
		return (f,ax)


def getBagOfWords(df, column_number,  additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10)):
	'''
	returns a "bag of words" the "dictionary" and a list of processed answers from the input dataframe that contains a column filled 
	with rows of answers for a given question.  Additional arguments allow for optional filtering of the data (as explained below).

	The "dictionary" contains a list of unique words in alphabetical order that appear in the answers

	The "bag of words" is a list of lists.  Each entry in the main list is for a single answer to the question (i.e., a single cell in the df answer column).  
	Within that list there is a list of tuples, where the first entry in a given tuple is the index of the word in the associated dictionary, and the second 
	entry is the number of occurences of that word in that answer.

	The final output, processed_answers, contains a list of the answer entries from df after running through the preprocess function.

	arguments
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, required) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming
	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (typical value may be 1e5)
	'''

	# convert the answers column to a list
	list_of_answers = df[df.columns[column_number]].tolist() 

	# preprocess each answer separately
	processed_answers = []
	for answer in list_of_answers:
		processed_answers.append(preprocess(answer, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem))
	processed_answers

	# convert to a "bag of words"
	dictionary = gensim.corpora.Dictionary(processed_answers)

	# filter
	dictionary.filter_extremes(no_below = no_below, no_above = no_above, keep_n = keep_n)

	# Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
	# words and how many times those words appear. Save this to 'bow_corpus'
	bow_corpus = [dictionary.doc2bow(doc) for doc in processed_answers]

	return (dictionary, bow_corpus, processed_answers)

def printDictionary(dictionary, N):
	'''
	print the first N entries in the "dictionary"

	arguments
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	N : (integer, required) anumber of entries to print
	'''
	count = 0
	for k, v in dictionary.iteritems():
		print(k, v)
		count += 1
		if (count >= N):
			break

def printBagOfWords(dictionary, bow_corpus, index):
	'''
	print the entries in the "bag of words" showing the word number, the word, and the number of times it appears in answer index

	arguments
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	bow_corpus : (list, required) the bag of words created using the dictionary, as outputted by getBagOfWords
	index : (integer, required) the index number of the answer that you want to display
	'''
	bow_answer = bow_corpus[index]

	for i in range(len(bow_answer)):
		print(f'Word {bow_answer[i][0]} ("{dictionary[bow_answer[i][0]]}") appears {bow_answer[i][1]} time.')

def runLDATopicModel(df, column_number, num_topics, passes = 20, workers = 1, additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10)):
	'''

	Run NLTK + gensim, Latent Dirichlet Allocation (LDA) algorithm, which uses unsupervised learning to extract the main topics 
	(i.e., a set of words) that occur in a collection of text samples.  This function uses the getBagOfWords function from above and then 
	runs the LDA topic modeling.  It returns the gensim "dictionary", "bag of words" and the LDA model.

	If num_topics is provides as a list of numbers, the output LDA model, perplexity and coherence values will also be lists (with each entry corresponding to the 
	number of topics at that index).

	The coherence measure uses multiple techniques, and is output as a list of [u_mass, c_v, c_uci, c_npmi] for each num_topics value.


	arguments
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, required) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	num_topics : (integer or list of integers, required) number(s) of topics to search for
	passes : (integer, optional) number of passes through the corpus during training
	workers : (integer, optional) number of worker processes to be used for parallelization.  The maximum value would be the number of real cores (not hyperthreads) - 1. 

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (typical value may be 1e5)




	-----
	Note: there are a lot of additional arguments that could be passed to LdaMulticore.
	documentation for gensim's lda muticore is here : https://radimrehurek.com/gensim/models/ldamulticore.html

	Possibly the most useful could be alpha and eta, hyperparameters that affect sparsity of the document-topic 
	and topic-word distributions. Currently, these are the default values (1/num_topics)

	- Alpha is the per document topic distribution.
		* High alpha: Every document has a mixture of all topics(documents appear similar to each other).
		* Low alpha: Every document has a mixture of very few topics

	- Eta is the per topic word distribution.
		* High eta: Each topic has a mixture of most words(topics appear similar to each other).
		* Low eta: Each topic has a mixture of few words.


	useful tutorials:
		- a good example notebook : https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb
		- NLTK and gensim tutorial: https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
		- NLTK and gensim tutorial : https://towardsdatascience.com/introduction-to-nlp-part-5b-unsupervised-topic-model-in-python-ab04c186f295

	'''


	# LDA mono-core -- fallback code in case LdaMulticore throws an error
	# lda_model = gensim.models.LdaModel(bow_corpus, 
	#                                    num_topics = num_topics, 
	#                                    id2word = dictionary,                                    
	#                                    passes = passes)


	# get the bag of words
	dictionary, bow_corpus, processed_answers = getBagOfWords(df, column_number,  additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n)

	if (type(num_topics) != list and type(num_topics) != np.ndarray):
		num_topics = [num_topics]

	models = []
	p = []
	c = []
	for n in num_topics:

		# Train your lda model using gensim.models.LdaMulticore 
		lda_model =  gensim.models.LdaMulticore(
			bow_corpus, 
			num_topics = n, 
			id2word = dictionary,  
			passes = passes,
			workers = workers
		)


		# Compute Perplexity
		# a measure of how good the model is. lower the better.
		perplexity = lda_model.log_perplexity(bow_corpus)  

		# Compute Coherence Score using all the different algorithms
		# 'u_mass'
		coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = 'u_mass', processes = workers)
		u_mass = coherence_model.get_coherence()

		# 'c_v'
		coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = 'c_v', processes = workers)
		c_v = coherence_model.get_coherence()

		# 'c_uci'
		coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = 'c_uci', processes = workers)
		c_uci = coherence_model.get_coherence()

		# 'c_npmi'
		coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = 'c_npmi', processes = workers)
		c_npmi = coherence_model.get_coherence()


		models.append(lda_model)
		p.append(perplexity)
		c.append([u_mass, c_v, c_uci, c_npmi])

		print(f'for {n} topics, perplexity = {perplexity:.3f} and coherence = [{u_mass:.3f}, {c_v:.3f}, {c_uci:.3f}, {c_npmi:.3f}]')

	if (len(num_topics) == 1):
		models = models[0]
		p = p[0]
		c = c[0]

	return dictionary, bow_corpus, models, p, np.array(c)


def printTopicModel(lda_model):
	'''
	For each topic, print the words occuring in that topic and its relative weight.

	arguments
	lda_model : (gensim ldamodel object, required) the LDA model object outputted by runTopicModel
	'''

	for idx, topic in lda_model.print_topics():
		print(f'Topic: {idx}\nWords: {topic}\n')


def plotLDAMetrics(num_topics, coherence, perplexity, best_index = None):
	'''
	generate line plots showing the coherence (top) and perplexity (bottom) scores for each model as a function of num_topics.
	returns the plot and axis objects

	arguments
	num_topcs : (list of integers, required) contains the numbers of topics with corresponding LDA models
	coherence : (list of floats, required) the coherence scores for each model
	perplexity : (list of floats, required) the perplexity scores for each model
	best_index : (integer, optional) the index of the best model (for a vertical dashed line)
	'''

	f, (ax1, ax2) = plt.subplots(2,1, figsize = (5, 5), sharex = True)

	cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
	for i,c in enumerate(cvals):
		cc = coherence[:,i]
		cn = (cc - min(cc))/(max(cc) - min(cc))
		ax1.plot(num_topics, cn, label = c)
	ax1.legend()
	ax2.plot(num_topics, perplexity)

	ax2.set_xlabel('Number of Topics')
	ax1.set_ylabel('Normalized Coherence')
	ax2.set_ylabel('log(Perplexity)')

	# choose the best model and plot a vertical line
	if (best_index is not None):
		ax1.axvline(x = num_topics[best_index], color ='gray', linestyle ='dashed')
		ax2.axvline(x = num_topics[best_index], color ='gray', linestyle ='dashed')

	plt.subplots_adjust(hspace = 0.0)

	return (f, (ax1, ax2))