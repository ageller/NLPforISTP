import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

import unicodedata
import re

import nltk
from nltk.corpus import stopwords

import gensim

import spacy

'''
useful tutorials:
- n-grams using NLTK : https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460
- a good example notebook for topic modeling : https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb
- NLTK and gensim tutorial on topic modeling : https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
- NLTK and gensim tutorial on topic modeling : https://towardsdatascience.com/introduction-to-nlp-part-5b-unsupervised-topic-model-in-python-ab04c186f295
- text summarization using spacy : https://medium.com/analytics-vidhya/text-summarization-using-spacy-ca4867c6b744 and notebook : https://github.com/kamal2230/text-summarization/blob/master/Summarisation_using_spaCy.ipynb
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

def getStringOfWords(df, column_number = 1):
	'''
	returns a string of words from a dataframe that contains a column filled with rows of answers for a given question

	arguments
	df : (pandas DataFrame, required) data set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
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
	_ = ax.set_xlabel('# of Occurences')

	if (ax is None):
		return (f,ax)


def getBagOfWords(df, column_number = 1,  additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10)):
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
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming
	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (a high value would keep everything)
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

def runLDATopicModel(df, column_number = 1, num_topics = 3, passes = 20, workers = 1, additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None):
	'''

	Run NLTK + gensim, Latent Dirichlet Allocation (LDA) algorithm, which uses unsupervised learning to extract the main topics 
	(i.e., a set of words) that occur in a collection of text samples.  This function uses the getBagOfWords function from above and then 
	runs the LDA topic modeling.  It returns the gensim "dictionary", "bag of words" and the LDA model.

	If num_topics is provides as a list of numbers, the output LDA model, perplexity and coherence values will also be lists (with each entry corresponding to the 
	number of topics at that index).

	The coherence measure uses multiple techniques, and is output as a list of [u_mass, c_v, c_uci, c_npmi] for each num_topics value.


	arguments
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	num_topics : (integer or list of integers, optional) number(s) of topics to search for
	passes : (integer, optional) number of passes through the corpus during training
	workers : (integer, optional) number of worker processes to be used for parallelization.  The maximum value would be the number of real cores (not hyperthreads) - 1. 

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (a very high value would keep everything)

	random_seed : ({np.random.RandomState, int}, optional) Either a randomState object or a seed to generate one. Useful for reproducibility. Note that results can still vary due to non-determinism in OS scheduling of the worker processes.


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
	cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
	c = {}
	for cv in cvals:
		c[cv] = []

	print('\n')
	for n in num_topics:

		# Train your lda model using gensim.models.LdaMulticore 
		lda_model =  gensim.models.LdaMulticore(
			bow_corpus, 
			num_topics = n, 
			id2word = dictionary,  
			passes = passes,
			workers = workers,
			random_state = random_seed
		)


		# Compute Perplexity
		# a measure of how good the model is. lower the better.
		perplexity = lda_model.log_perplexity(bow_corpus)  

		# Compute Coherence Score using all the different algorithms
		for cv in cvals:
			coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = cv, processes = workers)
			c[cv].append(coherence_model.get_coherence())

		models.append(lda_model)
		p.append(perplexity)

		cprint = ', '.join([f'{key}: {c[key][-1]:.3f}' for key in c.keys()])
		print(f'for {n} topics, perplexity = {perplexity:.3f} and coherence = {{{cprint}}}')

	if (len(num_topics) == 1):
		models = models[0]
		p = p[0]
		for key in c.keys():
			c[key] = c[key][0]

	print('\n')
	return dictionary, bow_corpus, models, p, c


def printLDATopicModel(lda_model):
	'''
	For each topic, print the words occuring in that topic and its relative weight.

	arguments
	lda_model : (gensim ldamodel object, required) the LDA model object outputted by runLDATopicModel
	'''

	for i, topic in lda_model.print_topics():
		print(f'********** Topic {i} **********')
		print(topic,'\n')


def plotLDAMetrics(num_topics, coherence, perplexity, best_index = None, colors = [None]):
	'''
	generate line plots showing the coherence (top) and perplexity (bottom) scores for each model as a function of num_topics.
	returns the plot and axis objects

	arguments
	num_topcs : (list of integers, required) contains the numbers of topics with corresponding LDA models
	coherence : (dict of lists, required) the coherence scores for each model where each dict entry has the type of coherence as the key
	perplexity : (list of floats, required) the perplexity scores for each model
	best_index : (integer, optional) the index of the best model (for a vertical dashed line)
	colors : (list, optional) define the colors for the plot; must be same length as the number of coherence values
	'''

	if (colors[0] is None):
		colors = ['C' + str(i) for i in range(len(coherence.keys()))]

	f, (ax1, ax2) = plt.subplots(2,1, figsize = (5, 5), sharex = True)

	cmult = np.ones(len(num_topics))
	for i, cv in enumerate(coherence.keys()):
		cc = coherence[cv]
		cn = (cc - min(cc))/(max(cc) - min(cc))
		cmult *= cn
		ax1.plot(num_topics, cn, label = cv, color = colors[i])
	#cmult = (cmult - min(cmult))/(max(cmult) - min(cmult))
	ax1.plot(num_topics, cmult, label = "combined", color = 'black', linestyle = 'dotted')
	ax1.legend()
	ax2.plot(num_topics, perplexity, color = 'black')

	ax2.set_xlabel('Number of Topics')
	ax1.set_ylabel('Normalized Coherence')
	ax2.set_ylabel('log(Perplexity)')
	ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
	ax2.xaxis.set_major_locator(MaxNLocator(integer = True))

	# choose the best model and plot a vertical line
	if (best_index is not None):
		ax1.axvline(x = num_topics[best_index], color ='gray', linestyle ='dashed')
		ax2.axvline(x = num_topics[best_index], color ='gray', linestyle ='dashed')

	plt.subplots_adjust(hspace = 0.0)

	return (f, (ax1, ax2))


def getLDAProbabilities(lda_model, bow_corpus, df, column_number = 1):
	'''
	Gets the probabilities for topics for each answer to a given question (as defined by the column at "column_number" in df).  
	Returns a pandas DataFrame containing one row per answer and columns with the answer text and probabilities that they exist in each topic,
	and a final column with the topic at the highest probability in each row.

	arguments
	lda_model : (gensim ldamodel object, required) the LDA model object outputted by runLDATopicModel
	bow_corpus : (gensim bag of words corpus, required)the gensim bag of words corpus, outputted by runLDATopicModel
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	'''

	# Predict probabilities
	predictions = lda_model.get_document_topics(bow_corpus, minimum_probability = 0.0)

	# format this into a Pandas DataFrame
	p = np.array([prediction for prediction in predictions])
	num_topics = len(p[0])

	df_p = pd.DataFrame()
	df_p['answers'] = df[df.columns[column_number]]
	for i in range(num_topics):
		df_p['topic' + str(i)] = p[:,i,1]

	# for each answer, get the topic at maximum probability
	top = []
	top_p = []
	for index, row in df_p.iterrows():
		sub = row['topic0':'topic' + str(num_topics - 1)]
		imax = np.argmax(sub)
		top.append('topic' + str(imax))
		top_p.append(sub[imax])
	df_p['top_topic'] = top
	df_p['top_probability'] = top_p


	return df_p

def plotTopLDAProbabilitiesKDE(df_p, topic_names = [None], colors = [None], bw_method = None):
	'''
	Generates histograms of the probabilities of the top topics for each answer.  Returns the figure and axis objects from pyplot.

	arguments
	df_p : (pandas DataFrame, required) output from getLDAProbabilities, needs a column named "top_topic" that points to another column with the probability values
	topic_names : (list, optional) define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
	colors : (list, optional) define the colors for the plot; must be same length as the number of topics
	bw_method (str, scalar or callable, optional) : the method used to calculate tehe estimator bandwidth (see scipy.stats.gaussian_kde documentation)
	'''


	if (topic_names[0] is None):
		topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

	if (colors[0] is None):
		colors = ['C' + str(i) for i in range(len(topic_names))]

	f,ax = plt.subplots()

	xs = np.linspace(0, 1, 200)
	for t,c in zip(topic_names, colors):
		# get a list of the probabilities for the rows that have this topic as the max topic
		df_tmp = df_p.loc[(df_p['top_topic'] == t)]
		# ax.hist(df_p[t], label = t, density = True, bins = bins)
		density = gaussian_kde(df_tmp[t], bw_method = bw_method)
		ax.plot(xs, density(xs), label = t, color = c)
		ax.fill_between(xs, density(xs), color = c, alpha = 0.3)

	ax.legend()
	ax.set_xlabel('probability')
	ax.set_ylabel('density')
	ax.set_xlim(0,1)
	ax.set_ylim(bottom = 0)

	return (f, ax)

def printBestLDATopicSentences(df_p, dictionary, lda_model, n_answers = 10, n_sentences = 2, topic_names = [None], show_answers = False):
	'''
	Find the top n_answers for a given topic based on the probability score (from getLDAProbabilities).  Then find the n_sentences that have the highest weight based on the topic keyword frequencies.  Print the topic words and the most relevant sentence(s).

	arguments
	df_p : (pandas DataFrame, required) output from getLDAProbabilities, needs a column named "answers" and additional columns with the topic probability values
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	lda_model : (gensim ldamodel object, required) the LDA model object outputted by runLDATopicModel
	n_answers : (integer, optional) number of answers to use for getting the best sentence (with the top n probabilities in that topic)
	n_sentences : (integer, optional) number of summary sentences to print
	topic_names : (list, optional) :define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
	show_answers : (boolean, options) whether to print the answer to the screen
	'''

	if (topic_names[0] is None):
		topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

	print('\n')
	for i, (tn, tw) in enumerate(zip(topic_names, lda_model.print_topics())):
		print(f'********** {tn} **********')
		print(tw[1],'\n')
		df_tmp = df_p.loc[df_p['top_topic'] == tn]
		indices = df_tmp[tn].nlargest(n_answers).index.values
		weighted = np.empty((0,2), "object")
		for j in indices:
			if (show_answers):
				print(f'topic probability = {df_tmp[tn][j]:.3f}')
				print(df_tmp['answers'][j], '\n')
			weighted = np.append(weighted, weightSentencesByLDATopic(df_tmp['answers'][j], dictionary, lda_model, i), axis = 0)

		print(f'Most relevant {n_sentences} sentence(s) from the top {n_answers} answers:\n')
		weighted_sorted = weighted[np.argsort(weighted[:, 1])[::-1]]
		for imax in range(n_sentences):
			print(f'Weight in topic = {weighted_sorted[imax][1]:.3f}')
			print(weighted_sorted[imax][0],'\n')

		print('\n')


def weightSentencesByLDATopic(text, dictionary, lda_model, topic_number):
	'''
	Weight the sentences in a given string of text by the words that are included in the given topic from the lda_model.
	Returns a numpy array with each entry as [sentence, weight].

	arguments
	text : (string, required) the text to be analyzed
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	lda_model : (gensim ldamodel object, required) the LDA model object outputted by runLDATopicModel
	topic_number : (integer, required) the topic to use for weighting
	'''

	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)

	available_words = np.array(list(dictionary.items()))[:,1]

	sentence_strength = {}
	for sentence in doc.sents:
		for word in sentence:
			if (word.text in available_words):
				prob = lda_model.get_term_topics(word.text)[topic_number][1]
				if sentence in sentence_strength.keys():
					sentence_strength[sentence] += prob
				else:
					sentence_strength[sentence] = prob

	return np.array(list(sentence_strength.items()), dtype = "object")



def runLSITopicModel(df, column_number = 1, num_topics = 3, additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None, workers = 1):
	'''

	Run NLTK + gensim, Latent Dirichlet Allocation (LDA) algorithm, which uses unsupervised learning to extract the main topics 
	(i.e., a set of words) that occur in a collection of text samples.  This function uses the getBagOfWords function from above and then 
	runs the LDA topic modeling.  It returns the gensim "dictionary", "bag of words" and the LDA model.

	If num_topics is provides as a list of numbers, the output LDA model, perplexity and coherence values will also be lists (with each entry corresponding to the 
	number of topics at that index).

	The coherence measure uses multiple techniques, and is output as a list of [u_mass, c_v, c_uci, c_npmi] for each num_topics value.


	arguments
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	num_topics : (integer or list of integers, optional) number(s) of topics to search for
	workers : (integer, optional) number of worker processes to be used for parallelization.  The maximum value would be the number of real cores (not hyperthreads) - 1. 

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (typical value may be 1e5)

	random_seed : ({np.random.RandomState, int}, optional) Either a randomState object or a seed to generate one. Useful for reproducibility. Note that results can still vary due to non-determinism in OS scheduling of the worker processes.


	-----
	Note: there are a lot of additional arguments that could be passed to LdaMulticore.
	documentation for gensim's lsi model is here : https://radimrehurek.com/gensim/models/lsimodel.html


	useful tutorials:
		- https://medium.com/@zeina.thabet/topic-modeling-with-lsi-lda-and-document-clustering-with-carrot2-part-1-5b1fbec737f6

	'''



	# get the bag of words
	dictionary, bow_corpus, processed_answers = getBagOfWords(df, column_number,  additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n)

	# # convert this to tfidf
	# tfidf = gensim.models.TfidfModel(bow_corpus)
	# corpus_tfidf = tfidf[bow_corpus]

	if (type(num_topics) != list and type(num_topics) != np.ndarray):
		num_topics = [num_topics]

	models = []
	p = []
	cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
	c = {}
	for cv in cvals:
		c[cv] = []

	print('\n')
	for n in num_topics:

		# Train your lsi model using gensim.models.LdaMulticore 
		lsi_model =  gensim.models.LsiModel(
			corpus = bow_corpus, #corpus_tfidf, 
			num_topics = n, 
			id2word = dictionary,  
			random_seed = random_seed
		)


		# Compute Coherence Score using all the different algorithms
		for cv in cvals:
			coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lsi_model, texts = processed_answers, corpus = bow_corpus, dictionary = dictionary, coherence = cv, processes = workers)
			c[cv].append(coherence_model.get_coherence())

		models.append(lsi_model)

		cprint = ', '.join([f'{key}: {c[key][-1]:.3f}' for key in c.keys()])
		print(f'for {n} topics coherence = {{{cprint}}}')

	if (len(num_topics) == 1):
		models = models[0]
		for key in c.keys():
			c[key] = c[key][0]

	print('\n')
	return dictionary, bow_corpus, models, c

def plotLSIMetrics(num_topics, coherence, best_index = None, colors = [None]):
	'''
	generate line plots showing the coherence (top) and perplexity (bottom) scores for each model as a function of num_topics.
	returns the plot and axis objects

	arguments
	num_topcs : (list of integers, required) contains the numbers of topics with corresponding LDA models
	coherence : (dict of lists, required) the coherence scores for each model where each dict entry has the type of coherence as the key
	best_index : (integer, optional) the index of the best model (for a vertical dashed line)
	colors : (list, optional) define the colors for the plot; must be same length as the number of coherence values
	'''

	if (colors[0] is None):
		colors = ['C' + str(i) for i in range(len(coherence.keys()))]

	f, ax1 = plt.subplots(figsize = (5, 3))

	cmult = np.ones(len(num_topics))
	for i, cv in enumerate(coherence.keys()):
		cc = coherence[cv]
		cn = (cc - min(cc))/(max(cc) - min(cc))
		cmult *= cn
		ax1.plot(num_topics, cn, label = cv, color = colors[i])
	#cmult = (cmult - min(cmult))/(max(cmult) - min(cmult))
	ax1.plot(num_topics, cmult, label = "combined", color = 'black', linestyle = 'dotted')
	ax1.legend()

	ax1.set_xlabel('Number of Topics')
	ax1.set_ylabel('Normalized Coherence')
	ax1.xaxis.set_major_locator(MaxNLocator(integer = True))

	# choose the best model and plot a vertical line
	if (best_index is not None):
		ax1.axvline(x = num_topics[best_index], color ='gray', linestyle ='dashed')


	return (f, ax1)

def getLSIVectors(lsi_model, bow_corpus, df, column_number = 1):
	'''
	Gets the sum of the coefficients of the vectors for topics for each answer to a given question (as defined by the column at "column_number" in df).  
	Returns a pandas DataFrame containing one row per answer and columns with the answer text and vectors that they exist in each topic,
	and a final column with the topic at the highest abs(vector) in each row.

	arguments
	lsi_model : (gensim lsimodel object, required) the LSI model object outputted by runLSITopicModel
	bow_corpus : (gensim bag of words corpus, required)the gensim bag of words corpus, outputted by runLDATopicModel
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	'''


	# calculate vectors
	# this returns the sum of the coefficient values for a given topic for each word in a given answer
	# but I'm not quite sure how to interpret this.  Does the best-matched topic have the largest abs(vector)? 
	vectors = lsi_model[bow_corpus]

	# get the number of topics
	num_topics = len(lsi_model.get_topics())

	# format this into a Pandas DataFrame (need to fix empty answers)
	p = []
	for v in vectors:
		check = [list(p_topic) for p_topic in v]
		if (len(check) < num_topics):
			check = [(i, 0.) for i in range(num_topics)]
		p.append(check)
	p = np.array(p)

	df_p = pd.DataFrame()
	df_p['answers'] = df[df.columns[column_number]]
	for i in range(num_topics):
		df_p['topic' + str(i)] = p[:,i,1]

	# for each answer, get the topic at maximum vector length
	top = []
	top_p = []
	for index, row in df_p.iterrows():
		sub = row['topic0':'topic' + str(num_topics - 1)]
		imax = np.argmax(np.abs(sub))
		top.append('topic' + str(imax))
		top_p.append(sub[imax])
	df_p['top_topic'] = top
	df_p['top_vector'] = top_p


	return df_p

def getLSIDistances(lsi_model, bow_corpus, df, column_number = 1):
	'''
	Gets the Euclidean distances for each answer to a given question (as defined by the column at "column_number" in df) from each topic in the model.  
	Returns a pandas DataFrame containing one row per answer and columns with the answer text and distances that they exist in each topic,
	and two final columns with the topic at the lowest distance in each row and that distance value.

	arguments
	lsi_model : (gensim lsimodel object, required) the LSI model object outputted by runLSITopicModel
	bow_corpus : (gensim bag of words corpus, required)the gensim bag of words corpus, outputted by runLDATopicModel
	df : (pandas DataFrame, required) ata set containing rows of responses to one (or more) question(s)
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
	'''

	# calculate the Euclidean distance between the sentence and topic vectors
	# note: this does not account for a word being stated more than once
	topic_vectors = lsi_model.get_topics()
	dists = []
	for bow in bow_corpus:
		dv = []
		for tindex, tvector in enumerate(topic_vectors):
			answer_vector = np.zeros(len(tvector))
			for i,w in enumerate(bow):
				answer_vector[w[0]] = tvector[w[0]]
			d = np.linalg.norm(tvector - answer_vector)
			dv.append([tindex, d])
		dists.append(dv)

	# calculate vectors
	# this returns the sum of the coefficient values for a given topic for each word in a given answer
	# but I'm not quite sure how to interpret this.  Does the best-matched topic have the largest abs(vector)? 
	vectors = lsi_model[bow_corpus]

	# get the number of topics
	num_topics = len(topic_vectors)

	# format this into a Pandas DataFrame
	p = np.array(dists)

	df_p = pd.DataFrame()
	df_p['answers'] = df[df.columns[column_number]]
	for i in range(num_topics):
		df_p['topic' + str(i)] = p[:,i,1]

	# for each answer, get the topic at maximum vector length
	top = []
	top_p = []
	for index, row in df_p.iterrows():
		sub = row['topic0':'topic' + str(num_topics - 1)]
		imin = np.argmin(sub)
		top.append('topic' + str(imin))
		top_p.append(sub[imin])
	df_p['top_topic'] = top
	df_p['top_distance'] = top_p


	return df_p

def plotTopLSIVectorsKDE(df_p, topic_names = [None], colors = [None], bw_method = None):
	'''
	Generates histograms of the vectors of the top topics for each answer.  Returns the figure and axis objects from pyplot.

	arguments
	df_p : (pandas DataFrame, required) output from getLSIVectors, needs a column named "top_topic" that points to another column with the vector values
	topic_names : (list, optional) define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
	colors : (list, optional) define the colors for the plot; must be same length as the number of topics
	bw_method (str, scalar or callable, optional) : the method used to calculate tehe estimator bandwidth (see scipy.stats.gaussian_kde documentation)
	'''


	if (topic_names[0] is None):
		topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

	if (colors[0] is None):
		colors = ['C' + str(i) for i in range(len(topic_names))]

	f,ax = plt.subplots()

	x = [row[row['top_topic']] for i,row in df_p.iterrows()]
	xs = np.linspace(min(x), max(x), 500) # picking a large range; not sure what the limits will always be
	for t,c in zip(topic_names, colors):
		# get a list of the probabilities for the rows that have this topic as the max topic
		df_tmp = df_p.loc[(df_p['top_topic'] == t)]
		# ax.hist(df_p[t], label = t, density = True, bins = bins)
		density = gaussian_kde(df_tmp[t], bw_method = bw_method)
		ax.plot(xs, density(xs), label = t, color = c)
		ax.fill_between(xs, density(xs), color = c, alpha = 0.3)

	ax.legend()
	ax.set_xlabel('distance')
	ax.set_ylabel('density')
	ax.set_xlim(min(xs), max(xs))
	ax.set_ylim(bottom = 0)

	return (f, ax)
	
def printBestLSITopicSentences(df_p, dictionary, lsi_model, n_answers = 10, n_sentences = 2, topic_names = [None], show_answers = False,  additional_stopwords = [''], wlen = 4, stem = True):
	'''
	Find the top n_answers for a given topic based on the distance score (from getLSIDistances).  Then find the n_sentences that have the lower distance from the topic vectors.  Print the topic words and the most relevant sentence(s).

	arguments
	df_p : (pandas DataFrame, required) output from getLSIDistnaces, needs a column named "answers" and additional columns with the topic probability values
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	lsi_model : (gensim lsimodel object, required) the LSI model object outputted by runLSITopicModel
	n_answers : (integer, optional) number of answers to use for getting the best sentence (with the top n probabilities in that topic)
	n_sentences : (integer, optional) number of summary sentences to print
	topic_names : (list, optional) :define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
	show_answers : (boolean, options) whether to print the answer to the screen

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	'''

	if (topic_names[0] is None):
		topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

	print('\n')
	for i, (tn, tw) in enumerate(zip(topic_names, lsi_model.print_topics())):
		print(f'********** {tn} **********')
		print(tw[1],'\n')
		df_tmp = df_p.loc[df_p['top_topic'] == tn]
		indices = df_tmp[tn].nsmallest(n_answers).index.values
		weighted = np.empty((0,2), "object")
		for j in indices:
			if (show_answers):
				print(f'topic distance = {df_tmp[tn][j]:.3f}')
				print('top topic = ', df_tmp['top_topic'][j])
				print(df_tmp['answers'][j], '\n')
			weighted = np.append(weighted, weightSentencesByLSITopic(df_tmp['answers'][j], dictionary, lsi_model, i, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem), axis = 0)

		print(f'Most relevant {n_sentences} sentence(s) from the top {n_answers} answers:\n')
		weighted_sorted = weighted[np.argsort(weighted[:, 1])]
		for imin in range(n_sentences):
			print(f'Distance from topic = {weighted_sorted[imin][1]:.3f}')
			print(weighted_sorted[imin][0],'\n')

		print('\n')


def weightSentencesByLSITopic(text, dictionary, lsi_model, topic_number,  additional_stopwords = [''], wlen = 4, stem = True):
	'''
	Weight the sentences in a given string of text by the words that are included in the given topic from the lda_model.
	Returns a numpy array with each entry as [sentence, weight].

	arguments
	text : (string, required) the text to be analyzed
	dictionary : (gensim dictionary object, required) a dictionary object as output from getBagOfWords
	lsi_model : (gensim lsimodel object, required) the LSI model object outputted by runLSITopicModel
	topic_number : (integer, required) the topic to use for weighting

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	'''

	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)

	available_words = np.array(list(dictionary.items()))[:,1]

	topic_vectors = lsi_model.get_topics()
	tvector = topic_vectors[topic_number]

	sentence_strength = {}
	for sentence in doc.sents:
		sentence_vector = np.zeros(len(tvector))
		text = preprocess(sentence.text, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)
		bow = dictionary.doc2bow(text)
		for i,w in enumerate(bow):
			sentence_vector[w[0]] = tvector[w[0]]
		d = np.linalg.norm(tvector - sentence_vector)
		sentence_strength[sentence] = d


	return np.array(list(sentence_strength.items()), dtype = "object")

def runNLPPipeline(filename, sheet = None, column_number = 1, additional_stopwords = [''], wlen = 3, stem = True, num_topics = 3, passes = 20, workers = 1, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None, coherence_method = 'c_v', topic_names = [None], c_colors = [None], p_colors = [None], bw_method = None, n_answers = 10, n_sentences = 2, show_answers = False, run_lda = True, run_lsi = True, run_ngrams = True):
	'''
	Run the full NLP analysis pipeline, including ngrams and LDA topic modeling

	arguments
	filename : (string, required) path to the file that stores the data, if sheet is supplied this is assumed to be an Excel file, otherwise it is assumes to be a csv file
	sheet : (string, optional) if supplying an Excel file, this gives the sheet name 
	column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)

	additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
	wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
	stem : (boolean, optional) whether to apply stemming

	num_topics : (integer or list of integers, optional) number(s) of topics to search for in LDA topic modeling
	passes : (integer, optional) number of passes through the corpus during training
	workers : (integer, optional) number of worker processes to be used for parallelization.  The maximum value would be the number of real cores (not hyperthreads) - 1. 
	random_seed : ({np.random.RandomState, int}, optional) Either a randomState object or a seed to generate one. Useful for reproducibility. Note that results can still vary due to non-determinism in OS scheduling of the worker processes.

	no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
	no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
	keep_n : (integer, optional) number of words to keep (typical value may be 1e5)

	coherence_method : (string, optional) the coherence measure to use when selecting the best number of topics, if set as "combined" then the coherence scores are normalized to be between 0 and 1 and then multiplied together and used for the best index

	topic_names : (list, optional) define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
	c_colors : (list, optional) define the colors for the coherence plot; must be same length as the number of coherence values 
	p_colors : (list, optional) define the colors for the topic probabilities plot; must be same length as the number of topics
	bw_method (str, scalar or callable, optional) : the method used to calculate tehe estimator bandwidth (see scipy.stats.gaussian_kde documentation)

	n_answers : (integer, optional) number of answers to use for getting the best sentence (with the top n probabilities in that topic)
	n_sentences : (integer, optional) number of summary sentences to print
	show_answers : (boolean, optional) whether to print the answer to the screen

	run_lda : (boolean, optional) whether to run the LDA analysis
	run_lsi : (boolean, optional) whether to run the LSI analysis
	run_ngrams : (boolean, optional) whether to run the n-grams analysis
	'''

	# read in the data
	print(f'  -- Reading data from file: "{filename}" --')
	if (sheet is None):
		df = pd.read_csv(filename)
		sheet = 'plot' # for naming the figures
	else:
		print(f'  -- Using sheet: "{sheet}"" --')
		df = pd.read_excel(filename, sheet)


	# get the bigrams and trigrams and create bar charts of the results
	if (run_ngrams):
		print(f'  -- Finding bigrams and trigrams --')
		# get a string of the words contained in all the answers from this DataFrame
		string_of_answers = getStringOfWords(df, column_number)

		# get the bigrams and trigrams
		bigrams = getNgrams(string_of_answers, 2, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)
		trigrams = getNgrams(string_of_answers, 3, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)

		# create a plot of the bigrams and trigrams
		fname = 'ngrams_' + sheet.replace(' ','') + '.png'
		print(f'  -- Saving ngrams figure to: "{fname}" --')
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
		N = 20
		plotNgrams(bigrams, N, ax = ax1)
		plotNgrams(trigrams, N, ax = ax2)
		_ = ax1.set_title(str(N) + ' Most Frequently Occuring Bigrams')
		_ = ax2.set_title(str(N) + ' Most Frequently Occuring Trigrams')
		plt.subplots_adjust(wspace = 0.6, left = 0.15, right = 0.99, top = 0.95, bottom = 0.07)
		f.savefig(fname, bbox_inches = 'tight')

	####################################
	# run the LDA topic model
	lda_model = None
	lda_perplexity = None
	lda_coherence = None 
	lda_df_p = None
	lda_best_index = None
	if (run_lda):
		print('  -- Running LDA topic model --')
		dictionary, bow_corpus, lda_model, lda_perplexity, lda_coherence = runLDATopicModel(df, column_number = column_number, num_topics = num_topics, passes = passes, workers = workers, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n, random_seed = random_seed)

		# choose the index of the best model by selecting the maximum coherence score
		if (coherence_method == "combined"):
			cmult = np.ones(len(num_topics))
			for i, cv in enumerate(lda_coherence.keys()):
				cc = lda_coherence[cv]
				cn = (cc - min(cc))/(max(cc) - min(cc))
				cmult *= cn
			lda_best_index = np.argmax(cmult)
		else:
			lda_best_index = np.argmax(lda_coherence[coherence_method])
		
		print(f'  -- The best LDA model has {num_topics[lda_best_index]} topics, using the "{coherence_method}" coherence method')

		# plot the results
		# higher coherence is better
		# lower perplexity is better
		fname = 'LDAmetrics_' + sheet.replace(' ','') + '.png'
		print(f'  -- Saving LDA metrics plot to: "{fname}" --')
		f, (ax1, ax2) = plotLDAMetrics(num_topics, lda_coherence, lda_perplexity, lda_best_index, colors = c_colors)
		f.savefig(fname, bbox_inches = 'tight')

		# calculate the probabilities for each answer being in each topic
		print(f'  -- Calculating probabilities of each LDA topic for each answer --')
		lda_df_p = getLDAProbabilities(lda_model[lda_best_index], bow_corpus, df, column_number = column_number)

		# plot a KDE of the probability distributions for each topic
		fname = 'LDAprobabilities_' + sheet.replace(' ','') + '.png'
		print(f'  -- Saving plot of KDE of top LDA topic probabilities to: "{fname}" --')
		f, ax = plotTopLDAProbabilitiesKDE(lda_df_p, topic_names = topic_names, bw_method = bw_method, colors = p_colors)
		f.savefig(fname, bbox_inches = 'tight')

		# print the answers that have the maximum probability for each topic
		print(f'  -- Printing summary information for each LDA topic --')
		printBestLDATopicSentences(lda_df_p, dictionary, lda_model[lda_best_index], n_answers = n_answers, n_sentences = n_sentences, topic_names = topic_names, show_answers = show_answers)


	###################################
	# run the LSI topic model
	lsi_model = None
	lsi_coherence = None
	lsi_df_d = None
	lsi_best_index = None
	if (run_lsi):
		print('  -- Running LSI topic model --')
		dictionary, bow_corpus, lsi_model, lsi_coherence = runLSITopicModel(df, column_number = column_number, num_topics = num_topics,  workers = workers, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n, random_seed = random_seed)

		# choose the index of the best model by selecting the maximum coherence score
		if (coherence_method == "combined"):
			cmult = np.ones(len(num_topics))
			for i, cv in enumerate(lsi_coherence.keys()):
				cc = lsi_coherence[cv]
				cn = (cc - min(cc))/(max(cc) - min(cc))
				cmult *= cn
			lsi_best_index = np.argmax(cmult)
		else:
			lsi_best_index = np.argmax(lsi_coherence[coherence_method])
		
		print(f'  -- The best LSI model has {num_topics[lsi_best_index]} topics, using the "{coherence_method}" coherence method')

		# plot the results
		# higher coherence is better
		fname = 'LSImetrics_' + sheet.replace(' ','') + '.png'
		print(f'  -- Saving LDA metrics plot to: "{fname}" --')
		f, ax = plotLSIMetrics(num_topics, lsi_coherence, lsi_best_index, colors = c_colors)
		f.savefig(fname, bbox_inches = 'tight')

		# calculate the probabilities for each answer being in each topic
		print(f'  -- Calculating LSI distances of each topic for each answer --')
		lsi_df_d = getLSIDistances(lsi_model[lsi_best_index], bow_corpus, df, column_number = column_number)

		# plot a KDE of the probability distributions for each topic
		fname = 'LSIdistances_' + sheet.replace(' ','') + '.png'
		print(f'  -- Saving plot of KDE of top LSI topic distances to: "{fname}" --')
		f, ax = plotTopLSIVectorsKDE(lsi_df_d, topic_names = topic_names, bw_method = bw_method, colors = p_colors)
		f.savefig(fname, bbox_inches = 'tight')

		# print the answers that have the maximum probability for each topic
		print(f'  -- Printing summary information for each topic --')
		printBestLSITopicSentences(lsi_df_d, dictionary, lsi_model[lsi_best_index], n_answers = n_answers, n_sentences = n_sentences, topic_names = topic_names, show_answers = show_answers, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)


	return {
			"dictionary":dictionary, 
			"bow_corpus":bow_corpus, 
			"lda":{
				"model":lda_model,
				"perplexity":lda_perplexity,
				"coherence":lda_coherence, 
				"df_p":lda_df_p, 
				"best_index":lda_best_index,
			},
			"lsi":{
				"model": lsi_model, 
				"coherence":lsi_coherence, 
				"df_d": lsi_df_d, 
				"best_index":lsi_best_index
			}
	    }
