import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from kneed import KneeLocator

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

def runLDATopicModel(df, column_number = 1, num_topics = 3, passes = 20, workers = 1, additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None, use_tfidf = False, cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']):
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
    use_tfidf : (boolean, optional) whether to convert the bow corpus to tf-idf, default False

    additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
    wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
    stem : (boolean, optional) whether to apply stemming

    no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
    no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
    keep_n : (integer, optional) number of words to keep (a very high value would keep everything)

    random_seed : ({np.random.RandomState, int}, optional) Either a randomState object or a seed to generate one. Useful for reproducibility. Note that results can still vary due to non-determinism in OS scheduling of the worker processes.

    cvals : (list, optional) list of strings defining the coherence metrics to use

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

    use_corpus = bow_corpus
    if (use_tfidf):
        # convert this to tfidf
        print('\n      *** using TF-IDF method ***')
        tfidf = gensim.models.TfidfModel(bow_corpus)
        use_corpus = tfidf[bow_corpus]
    
    if (type(num_topics) != list and type(num_topics) != np.ndarray):
        num_topics = [num_topics]

    models = []
    p = []
    c = {}
    for cv in cvals:
        c[cv] = []

    print('\n')
    for n in num_topics:

        # Train your lda model using gensim.models.LdaMulticore 
        lda_model =  gensim.models.LdaMulticore(
            use_corpus, 
            num_topics = n, 
            id2word = dictionary,  
            passes = passes,
            workers = workers,
            random_state = random_seed
        )

        # Compute Perplexity
        # a measure of how good the model is. lower the better.
        perplexity = lda_model.log_perplexity(use_corpus)  

        # Compute Coherence Score using all the different algorithms
        for cv in cvals:
            coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lda_model, texts = processed_answers, corpus = use_corpus, dictionary = dictionary, coherence = cv, processes = workers)
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
    return dictionary, use_corpus, models, p, c


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
        cc = np.array(coherence[cv])
        # if (cv in ['c_uci','u_mass', 'c_npmi']):
        #     # these measurements are in the log
        #     cc = 10.**cc
        cn = (cc - min(cc))/(max(cc) - min(cc))
        if (cv in ['u_mass']):
            # take the inverse for the combined value -- might be necessary for others as well
            cmult *= (1. - cn)
        else:
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

def plotTopLDAProbabilitiesKDE(df_p, topic_names = [None], colors = [None], bw_method = None, n_prob = 100):
    '''
    Generates histograms of the top n_prob probabilities of each topic.  Returns the figure and axis objects from pyplot.

    arguments
    df_p : (pandas DataFrame, required) output from getLDAProbabilities, needs a column named "top_topic" that points to another column with the probability values
    topic_names : (list, optional) define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
    colors : (list, optional) define the colors for the plot; must be same length as the number of topics
    bw_method : (str, scalar or callable, optional) the method used to calculate tehe estimator bandwidth (see scipy.stats.gaussian_kde documentation)
    n_prob : (integer, optional) number of elements to use for generating the KDE of the topic probabilities
    '''


    if (topic_names[0] is None):
        topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

    if (colors[0] is None):
        colors = ['C' + str(i) for i in range(len(topic_names))]

    f,ax = plt.subplots()

    xs = np.linspace(0, 1, 200)
    for t,c in zip(topic_names, colors):
        # # get a list of the probabilities for the rows that have this topic as the max topic
        # df_tmp = df_p.loc[(df_p['top_topic'] == t)]
        # # ax.hist(df_p[t], label = t, density = True, bins = bins)
        # if (len(df_tmp[t]) > 3):
        #     density = gaussian_kde(df_tmp[t], bw_method = bw_method)
        #     ax.plot(xs, density(xs), label = t, color = c)
        #     ax.fill_between(xs, density(xs), color = c, alpha = 0.3)
        arr = np.array(df_p[t])
        sorted_arr = np.sort(arr)[::-1]
        density = gaussian_kde(sorted_arr[0:n_prob], bw_method = bw_method)
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
        # df_tmp = df_p.loc[df_p['top_topic'] == tn]
        df_tmp = df_p.copy()
        indices = df_tmp[tn].nlargest(n_answers).index.values
        weighted = np.empty((0,2), "object")
        for j in indices:
            if (show_answers):
                print(f'topic probability = {df_tmp[tn][j]:.3f}')
                print(df_tmp['answers'][j], '\n')
            txt = df_tmp['answers'][j]

            # I don't know why this sometimes returns a Series, but this should hopefully fix it if/when that happens
            if isinstance(txt, (pd.DataFrame, pd.Series)):
                txt = txt.values
            if (isinstance(txt, (list, np.ndarray))):
                txt = ' '.join(txt)

            weighted = np.append(weighted, weightSentencesByLDATopic(txt, dictionary, lda_model, i), axis = 0)

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
        sentence_strength[sentence] = 0.
        for word in sentence:
            if (word.text in available_words):
                try:
                    prob = lda_model.get_term_topics(word.text)[topic_number][1]
                except:
                    prob = 0
                if sentence in sentence_strength.keys():
                    sentence_strength[sentence] += prob

    return np.array(list(sentence_strength.items()), dtype = "object")



def runLSITopicModel(df, column_number = 1, num_topics = 3, additional_stopwords = [''], wlen = 3, stem = True, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None, workers = 1, use_tfidf = False, cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']
):
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
    use_tfidf : (boolean, optional) whether to convert the bow corpus to tf-idf before running the LSI topic model, default False

    additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
    wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
    stem : (boolean, optional) whether to apply stemming

    no_below : (integer, optional) filtering to remove words that apper less than no_below times (typical value may be 15)
    no_above : (float, optional) filtering to remove words that appear in more than no_above (fraction) of all documents (tpyical value may be 1; don't use)
    keep_n : (integer, optional) number of words to keep (typical value may be 1e5)

    random_seed : ({np.random.RandomState, int}, optional) Either a randomState object or a seed to generate one. Useful for reproducibility. Note that results can still vary due to non-determinism in OS scheduling of the worker processes.

    cvals : (list, optional) list of strings defining the coherence metrics to use


    -----
    Note: there are a lot of additional arguments that could be passed to LdaMulticore.
    documentation for gensim's lsi model is here : https://radimrehurek.com/gensim/models/lsimodel.html


    useful tutorials:
        - https://medium.com/@zeina.thabet/topic-modeling-with-lsi-lda-and-document-clustering-with-carrot2-part-1-5b1fbec737f6

    '''



    # get the bag of words
    dictionary, bow_corpus, processed_answers = getBagOfWords(df, column_number,  additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n)

    use_corpus = bow_corpus
    if (use_tfidf):
        # convert this to tfidf
        print('\n** using TF-IDF method **')
        tfidf = gensim.models.TfidfModel(bow_corpus)
        use_corpus = tfidf[bow_corpus]

    if (type(num_topics) != list and type(num_topics) != np.ndarray):
        num_topics = [num_topics]

    models = []
    p = []
    c = {}
    for cv in cvals:
        c[cv] = []

    print('\n')
    for n in num_topics:

        # Train your lsi model using gensim.models.LdaMulticore 
        lsi_model =  gensim.models.LsiModel(
            corpus = use_corpus,
            num_topics = n, 
            id2word = dictionary,  
            random_seed = random_seed
        )
        
        # Compute Coherence Score using all the different algorithms
        for cv in cvals:
            coherence_model = gensim.models.coherencemodel.CoherenceModel(model = lsi_model, texts = processed_answers, corpus = use_corpus, dictionary = dictionary, coherence = cv, processes = workers)
            c[cv].append(coherence_model.get_coherence())

        models.append(lsi_model)

        cprint = ', '.join([f'{key}: {c[key][-1]:.3f}' for key in c.keys()])
        print(f'for {n} topics coherence = {{{cprint}}}')

    if (len(num_topics) == 1):
        models = models[0]
        for key in c.keys():
            c[key] = c[key][0]

    print('\n')
    return dictionary, use_corpus, models, c

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
        cc = np.array(coherence[cv])
        # if (cv in ['c_uci','u_mass', 'c_npmi']):
        #     # these measurements are in the log
        #     cc = 10.**cc
        cn = (cc - min(cc))/(max(cc) - min(cc))
        if (cv in ['u_mass']):
            # take the inverse for the combined value -- might be necessary for others as well
            cmult *= (1. - cn)
        else:
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

def plotTopLSIVectorsKDE(df_p, topic_names = [None], colors = [None], bw_method = None, n_prob = 100):
    '''
    Generates histograms of the vectors of the top topics for each answer.  Returns the figure and axis objects from pyplot.

    arguments
    df_p : (pandas DataFrame, required) output from getLSIVectors, needs a column named "top_topic" that points to another column with the vector values
    topic_names : (list, optional) define the topic names. If undefined, then the code assumes that all topics will have a name beginning with the word "topic"
    colors : (list, optional) define the colors for the plot; must be same length as the number of topics
    bw_method : (str, scalar or callable, optional) the method used to calculate tehe estimator bandwidth (see scipy.stats.gaussian_kde documentation)
    n_prob : (integer, optional) number of elements to use for generating the KDE of the topic probabilities

    '''


    if (topic_names[0] is None):
        topic_names = [x for x in df_p.columns.values if x.startswith('topic')]

    if (colors[0] is None):
        colors = ['C' + str(i) for i in range(len(topic_names))]

    f,ax = plt.subplots()

    x = [row[row['top_topic']] for i,row in df_p.iterrows()]
    xs = np.linspace(min(x), max(x), 500) # picking a large range; not sure what the limits will always be
    for t,c in zip(topic_names, colors):
        # # get a list of the probabilities for the rows that have this topic as the max topic
        # df_tmp = df_p.loc[(df_p['top_topic'] == t)]
        # # ax.hist(df_p[t], label = t, density = True, bins = bins)
        # if (len(df_tmp[t]) > 0):
        #     density = gaussian_kde(df_tmp[t], bw_method = bw_method)
        #     ax.plot(xs, density(xs), label = t, color = c)
        #     ax.fill_between(xs, density(xs), color = c, alpha = 0.3)
        arr = np.array(df_p[t])
        sorted_arr = np.sort(arr)[::-1]
        density = gaussian_kde(sorted_arr[0:n_prob], bw_method = bw_method)
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
        # df_tmp = df_p.loc[df_p['top_topic'] == tn]
        df_tmp = df_p.copy()
        indices = df_tmp[tn].nsmallest(n_answers).index.values
        weighted = np.empty((0,2), "object")
        for j in indices:
            if (show_answers):
                print(f'topic distance = {df_tmp[tn][j]:.3f}')
                print('top topic = ', df_tmp['top_topic'][j])
                print(df_tmp['answers'][j], '\n')
            txt = df_tmp['answers'][j]

            # I don't know why this sometimes returns a Series, but this should hopefully fix it if/when that happens
            if isinstance(txt, (pd.DataFrame, pd.Series)):
                txt = txt.values
            if (isinstance(txt, (list, np.ndarray))):
                txt = ' '.join(txt)

            weighted = np.append(weighted, weightSentencesByLSITopic(txt, dictionary, lsi_model, i, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem), axis = 0)

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


def getTFIDFvector(processed_answers_list, dictionary = None, ngram_range = (1,1), min_df = 1, lsa_n_components = 100):
    '''
    get the Term Frequency - Inverse Document Frequency vector for a list of sentences

    arguments
    processed_answers_list : (list of strings, required) contains a list of strings that have been preprocessed. Each entry in the list should be one full answer
    dictionary : (gensim dictionary object, options) output from getBagOfWords
    ngram_range : (tuple, optional) which ngrams are desired in this analysis
    min_df : (float, optional)  ignore terms that have a document frequency strictly lower than this value (1 to not ingore anything)
    lsa_n_components : (integer, optiona) number of components for the lsa model; from docs "For LSA, a value of 100 is recommended." (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

    '''
    if (dictionary is None):
        tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = ngram_range, min_df = min_df)
    else:
        vocab = [v for k, v in dictionary.iteritems()]
        tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = ngram_range, min_df = min_df, vocabulary = vocab)
    tfidf_vector = tfidf_vectorizer.fit_transform(processed_answers_list)

    
    lsa_vectorizer = make_pipeline(TruncatedSVD(n_components = lsa_n_components), Normalizer(copy = False))
    lsa_vector = lsa_vectorizer.fit_transform(tfidf_vector)

    return tfidf_vectorizer, tfidf_vector, lsa_vectorizer, lsa_vector

def plotKmeanMetrics(sse, elbow = None):
    '''
    plot the the sum of squared errors (SSE) for the K-means analysis

    arguments
    sse : (dict, required) dictionary where keys contain the number of clusters (k value) and values contain the SSE results
    elbow : (float, optional) the elbow value in the sse array
    '''

    f,ax = plt.subplots()
    if (elbow is not None):
        ax.axvline(elbow, linestyle = '--', color = 'gray')
    ax.plot(list(sse.keys()), list(sse.values()), '-o', color = 'k')
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('SSE')

    return f,ax

def runKmeans(tfidf_vector, tfidf_vectorizer, df, num_clusters = 5, column_number = 1, num_words_for_label = 5, n_init = 5, lsa_vectorizer = None, lsa_vector = None, use_lsa = False, random_seed = None):
    '''
    run the k-means clustering analysis using a TF-IDF vector

    arguments
    tfidf_vector : (matrix, required) the TF-IDF vector matrix resulting from getTFIDFvector
    tfidf_vectorizer : (TfidfVectorizer, required) the vectorizer object resulting from getTFIDFvector
    df : (pandas dataframe, required) contains the answers
    num_clusters : (integer or array of integers, optional) the number of clusters to test; if array-like then the best number is found using the elbow method
    column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)
    num_words_for_label : (integer, optional) the number of words to keep for the topic label
    n_init : (integer, optional) number of k-means initializations steps (then chooses best), for lsa use 1 otherwise use 5
    lsa_vectorizer : (lsa vectorizer, optional) from getTFIDFvector
    lsa_vector : (array, optional) getTFIDFvector
    use_lsa : (boolean, optional) whether or not to use the lsa vector
    random_seed : (integer, optional) random seed
    '''

    vector = tfidf_vector
    if (use_lsa):
        vector = lsa_vector
        print('      *** using LSA dimensional reduction ***')

    if isinstance(num_clusters,(list, pd.core.series.Series, np.ndarray)) :
        #Test increments of clusters using elbow method
        sse = {}
        homogeneity = {}
        completeness = {}
        v_measure = {}
        adjusted_rand = {}
        silhouette = {}
        km_models = {}
        for k in num_clusters:
            km = KMeans(n_clusters = k, max_iter = 1000, n_init = n_init, random_state = random_seed)
            kmeans = km.fit(vector)
            km_models[k] = km
            sse[k] = kmeans.inertia_
            # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
            # but require a ground truth... for now we don't have that
            # homogeneity[k] = metrics.homogeneity_score(labels, km.labels_)
            # completeness[k] = metrics.completeness_score(labels, km.labels_)
            # v_measure[k] = metrics.v_measure_score(labels, km.labels_)
            # adjusted_rand[k] = metrics.adjusted_rand_score(labels, km.labels_)
            # silhouette[k] = metrics.silhouette_score(vector, km.labels_, sample_size = 2000)
        
        # find the elbow in this curve
        kneedle = KneeLocator(list(sse.keys()), list(sse.values()), S = 0.0, curve = "convex", direction = "decreasing", online = False, interp_method = "interp1d")

        n_clusters = kneedle.elbow
    else:
        n_clusters = num_clusters
        sse = None
        km_models = {n_clusters : KMeans(n_clusters = n_clusters, max_iter = 1000, n_init = n_init)}

    # run a final time with the optimal number of clusters
    km_models[n_clusters].fit(vector)

    result = pd.concat([df, 
                        pd.DataFrame(tfidf_vector.toarray(),
                                     columns = tfidf_vectorizer.get_feature_names_out()
                                    )
                       ],axis=1)

    result['cluster'] = km_models[n_clusters].predict(vector)


    # Label each cluster with the word(s) that all of its entries have in common
    clusters = np.sort(result['cluster'].unique())
    labels = []

    # from here : https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
    # but this doesn't seem to make sense with the sentences I'm picking out...
    if (use_lsa):
        original_space_centroids = lsa_vectorizer[0].inverse_transform(km_models[n_clusters].cluster_centers_)               
    else:
        original_space_centroids = km_models[n_clusters].cluster_centers_

    terms = tfidf_vectorizer.get_feature_names_out() 
    for i in range(len(clusters)):
        centroid_words = []
        # take the largest values along the term axes
        dist = original_space_centroids[i]
        order_dist = dist.argsort()[::-1]
        for ind in order_dist[:num_words_for_label]:
            centroid_words.append(terms[ind])
        labels.append(', '.join(centroid_words))

    # # old method, counting most frequent words (and matches my sentence method below)
    # for i in range(len(clusters)):
    #     subset = result[result['cluster'] == clusters[i]]
    #     exclude = [result.columns[j] for j in range(column_number + 1)] + ['cluster']
    #     subset_words = subset.drop(exclude, axis = 1)

    #     # count the number of times each word appears and take the top N
    #     count = subset_words.astype(bool).sum(axis = 0).sort_values(ascending = False)
    #     words = ', '.join(count[0:num_words_for_label].index)
    #     print("original", i, words)
    #     labels.append(words)


    labels_table = pd.DataFrame(zip(clusters,labels), columns=['cluster','label'])#.sort_values('cluster')
    # result_labelled = pd.merge(result,labels_table,on = 'cluster',how = 'left')

    return km_models, sse, homogeneity, completeness, v_measure, adjusted_rand, silhouette, n_clusters, result, labels_table 

def printBestKmeansSentences(kmeans_model, tfidf_vectorizer, tfidf_vector, labels_table, results_table, n_answers = 10, n_sentences = 2, column_number = 1, additional_stopwords = [''], wlen = 3, stem = True, lsa_vectorizer = None, lsa_vector = None, use_lsa = False):
    '''
    print the most relevant sentences for each cluster

    arguments
    kmeans_model : (scikit learn Kmeans model object, required) as output from runKmeans
    tfidf_vectorizer : (TfidfVectorizer, required) the vectorizer object resulting from getTFIDFvector
    tfidf_vector : (matrix, required) the TF-IDF vector matrix resulting rfom getTFIDFvector
    labels_table : (pandas DataFrame, required) DataFrame containing the labels for each cluster, e.g., as output from runKmeans, expects columns of 'cluster' and 'label'
    results_table : (pandas DataFrame, required) DataFrame containing the results of the kmeans analysis, as output from runKmeans
    n_answers : (integer, optional) number of answers to use for getting the best sentence (with the top n probabilities in that topic)
    n_sentences : (integer, optional) number of summary sentences to print	
    column_number : (integer, optional) the column number for the specific question (supplying a number is easier because the column name is usualy the question text)

    additional_stopwords : (list of strings, optional) a list of strings containing words that should be removed from text, other than the standard stopwords
    wlen : (integer, optional) minimum word length (any words < wlen will be excluded)
    stem : (boolean, optional) whether to apply stemming

    lsa_vectorizer : (lsa vectorizer, optional) from getTFIDFvector
    lsa_vector : (array, optional) getTFIDFvector
    use_lsa : (boolean, optional) whether or not to use the lsa vector
    '''

    # get sentences from the closest answers to the cluster
    if (use_lsa):
        dist = kmeans_model.transform(lsa_vector)
    else:
        dist = kmeans_model.transform(tfidf_vector)

    # get a list of rows to remove that have <= 1 word in the tf-idf matrix
    # (the results are tied to the tf-idf regardless of using lsa)
    row_sums = tfidf_vector.toarray().astype(bool).sum(axis = 1)
    result_prune = results_table[row_sums > 1]
    dist_prune = dist[row_sums > 1]

    nlp = spacy.load('en_core_web_sm')

    clusters = result_prune['cluster'].unique()
    for i in range(len(clusters)):
        print(f'********** Cluster {i} **********')
        print(labels_table.loc[labels_table['cluster'] == i]['label'].values)
        print('\n')
        
        # I need to check that we have enough sentences if we're using lsa, since kmeans will expect 100 "columns" and apparently this implementation of lsa will not create more columns than the number of rows (here sentences) it is given
        done = False
        trialN = 0
        n_answers_use = n_answers
        while (not done):
            nearest_answer_indices = np.argsort(dist_prune[:,i])[:n_answers_use]
            nearest_answers = result_prune.iloc[nearest_answer_indices]
            # print(nearest_answers['cluster'])
            
            # combine these into one long text
            combined_answers = nearest_answers[nearest_answers.columns[column_number]].str.cat(sep=' ')
            
            # split this into sentences
            doc = nlp(combined_answers)
            sentences = np.array([s.text for s in doc.sents])
            sentences_processed = []
            for s in doc.sents:
                text = preprocess(s.text, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem)
                sentences_processed.append(' '.join(text))
            done = True
            trialN += 1
            if (use_lsa and len(sentences_processed) < 100 and trialN < 100):
                done = False
                n_answers_use += 1

        if (trialN == 100):
            print('!!!!! could not perform lsa reduction, not enough sentences in data (you will need to reduce lsa_n_components !!!')

        # get a new TF-IDF vector for each of these answers and the distances
        tfidf_vector_sentences = tfidf_vectorizer.fit_transform(sentences_processed)
        if (use_lsa):
            lsa_vector_sentences = lsa_vectorizer.fit_transform(tfidf_vector_sentences)
            dist_sentences = kmeans_model.transform(lsa_vector_sentences)
        else:
            dist_sentences = kmeans_model.transform(tfidf_vector_sentences)

        # prune as above
        row_sums = tfidf_vector_sentences.toarray().astype(bool).sum(axis = 1)
        sentences_prune = sentences[row_sums > 1]
        dist_sentences_prune = dist_sentences[row_sums > 1]

        # get the nearest sentences
        nearest_sentence_indices = np.argsort(dist_sentences_prune[:,i])[:n_sentences]
        nearest_sentences = sentences_prune[nearest_sentence_indices]
        nearest_distances = dist_sentences_prune[:,i][nearest_sentence_indices]
        print(f'Most relevant {n_sentences} sentence(s) from the top {n_answers_use} answers:\n')
            
        for (s,d) in zip(nearest_sentences, nearest_distances):
            print(f'Distance from centroid = {d:.4}')
            print(s.strip(),'\n')


def runNLPPipeline(filename = None, df = None, sheet = None, column_number = 1, additional_stopwords = [''], wlen = 3, stem = True, num_topics = 3, passes = 20, workers = 1, no_below = 1, no_above = 1, keep_n = int(1e10), random_seed = None, coherence_method = 'c_v', topic_names = [None], c_colors = [None], p_colors = [None], bw_method = None, n_answers = 10, n_sentences = 2, kmeans_tfidf_ngram_range = (1,1), kmeans_tfidf_min_df = 1, kmeans_num_words_for_label = 5, kmeans_use_lsa = False, lsa_n_components = 100, show_answers = False, run_lda = True, run_lsi = True, run_kmeans = True, run_ngrams = True, use_tfidf = {'lsi':True, 'lda':False}, cvals = ['u_mass', 'c_v', 'c_uci', 'c_npmi']):
    '''
    Run the full NLP analysis pipeline, including ngrams and LDA topic modeling

    arguments
    filename : (string, optional) path to the file that stores the data, if sheet is supplied this is assumed to be an Excel file, otherwise it is assumes to be a csv file (if the filename is None, then df must be supplied)
    sheet : (string, optional) if supplying an Excel file, this gives the sheet name 
    df : (pandas DataFrame, optional) if not None, no file will be read in and this dataFrame will be used
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

    kmeans_tfidf_ngram_range : (tuple, optional) which ngrams are desired in this analysis
    kmeans_tfidf_min_df : (float, optional)  ignore terms that have a document frequency strictly lower than this value (1 to not ingore anything)
    kmeans_num_words_for_label : (integer, optional) the number of words to keep for the topic label
    kmeans_use_lsa : (boolean, optional) whether to use lsa to reduce dimensionality in kmeans analysis
    lsa_n_components : (integer, optiona) number of components for the lsa model; from docs "For LSA, a value of 100 is recommended." (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

    use_tfidf : (boolean dict, optional) whether to convert the bow corpus to tf-idf for LDA and/or LSA (k-means currently always uses tf-idf), default {'lsi':True, 'lda':False}

    cvals : (list, optional) list of strings defining the coherence metrics to use

    n_answers : (integer, optional) number of answers to use for getting the best sentence (with the top n probabilities in that topic)
    n_sentences : (integer, optional) number of summary sentences to print
    show_answers : (boolean, optional) whether to print the answer to the screen

    run_lda : (boolean, optional) whether to run the LDA analysis
    run_lsi : (boolean, optional) whether to run the LSI analysis
    run_kmeans : (boolean, optiona) whether to run the kmeans analysis
    run_ngrams : (boolean, optional) whether to run the n-grams analysis
    '''

    # need to Kmeans
    os.environ["OMP_NUM_THREADS"] = str(workers)
    
    # read in the data
    if (df is None):
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
        dictionary, bow_corpus, lda_model, lda_perplexity, lda_coherence = runLDATopicModel(df, column_number = column_number, num_topics = num_topics, passes = passes, workers = workers, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n, random_seed = random_seed, use_tfidf = use_tfidf['lda'], cvals = cvals)

        # choose the index of the best model by selecting the maximum coherence score
        if (coherence_method == "combined"):
            cmult = np.ones(len(num_topics))
            for i, cv in enumerate(lda_coherence.keys()):
                cc = np.array(lda_coherence[cv])
                # if (cv in ['c_uci','u_mass', 'c_npmi']):
                    # these measurements are in the log
                    # cc = 10.**cc
                cn = (cc - min(cc))/(max(cc) - min(cc))
                if (cv in ['u_mass']):
                    # take the inverse for the combined value -- might be necessary for others as well
                    cmult *= (1. - cn)
                else:
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
        dictionary, bow_corpus, lsi_model, lsi_coherence = runLSITopicModel(df, column_number = column_number, num_topics = num_topics,  workers = workers, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, no_below = no_below, no_above = no_above, keep_n = keep_n, random_seed = random_seed, use_tfidf = use_tfidf['lsi'], cvals = cvals)

        # choose the index of the best model by selecting the maximum coherence score
        if (coherence_method == "combined"):
            cmult = np.ones(len(num_topics))
            for i, cv in enumerate(lsi_coherence.keys()):
                cc = np.array(lsi_coherence[cv])
                # if (cv in ['c_uci','u_mass', 'c_npmi']):
                    # these measurements are in the log
                    # cc = 10.**cc
                cn = (cc - min(cc))/(max(cc) - min(cc))
                if (cv in ['u_mass']):
                    # take the inverse for the combined value -- might be necessary for others as well
                    cmult *= (1. - cn)
                else:
                    cmult *= cn
            lsi_best_index = np.argmax(cmult)
        else:
            lsi_best_index = np.argmax(lsi_coherence[coherence_method])
        
        print(f'  -- The best LSI model has {num_topics[lsi_best_index]} topics, using the "{coherence_method}" coherence method')

        # plot the results
        # higher coherence is better
        fname = 'LSImetrics_' + sheet.replace(' ','') + '.png'
        print(f'  -- Saving LSI metrics plot to: "{fname}" --')
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

    ###################################
    # run the k-means analysis
    km_model = None
    km_labels_table = None
    km_result_table = None
    km_sse = None
    km_n_clusters = None
    if (run_kmeans):
        print('  -- Running k-means clustering analysis --')
        dictionary, bow_corpus, processed_answers = getBagOfWords(df, column_number,  additional_stopwords = additional_stopwords)
        processed_answers_list = [' '.join(x) for x in processed_answers]
        tfidf_vectorizer, tfidf_vector, lsa_vectorizer, lsa_vector = getTFIDFvector(processed_answers_list, dictionary = dictionary, ngram_range = kmeans_tfidf_ngram_range, min_df = kmeans_tfidf_min_df, lsa_n_components = lsa_n_components)

        n_init = 5
        if (kmeans_use_lsa):
            n_init = 1

        km_models, km_sse, km_homogeneity, km_completeness, km_v_measure, km_adjusted_rand, km_silhouette,km_n_clusters, km_result_table, km_labels_table  = runKmeans(tfidf_vector, tfidf_vectorizer, df, num_clusters = num_topics,  column_number = column_number, num_words_for_label = kmeans_num_words_for_label, n_init = n_init, use_lsa = kmeans_use_lsa, lsa_vectorizer = lsa_vectorizer, lsa_vector = lsa_vector, random_seed = random_seed)

        print(f'  -- The best k-means model has {km_n_clusters} clusters.')

        # plot the results
        # higher coherence is better
        fname = 'kmeansMetrics_' + sheet.replace(' ','') + '.png'
        print(f'  -- Saving k-means metrics plot to: "{fname}" --')
        f, ax = plotKmeanMetrics(km_sse, km_n_clusters)
        f.savefig(fname, bbox_inches = 'tight')

        # print the summary information
        print(f'  -- Printing summary information for each cluster --')
        printBestKmeansSentences(km_models[km_n_clusters], tfidf_vectorizer, tfidf_vector, km_labels_table, km_result_table, n_answers = n_answers, n_sentences = n_sentences, column_number = column_number, additional_stopwords = additional_stopwords, wlen = wlen, stem = stem, lsa_vectorizer = lsa_vectorizer, lsa_vector = lsa_vector, use_lsa = kmeans_use_lsa)

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
            },
            "kmeans":{
                "model": km_model,
                "labels_table": km_labels_table,
                "result_table": km_result_table,
                "metrics": {
                    'sse': km_sse,
                    'homogeneity': km_homogeneity, 
                    'completeness': km_completeness,
                    'v_measure': km_v_measure, 
                    'adjusted_rand': km_adjusted_rand, 
                    'silhouette': km_silhouette,
                },
                "best_index": km_n_clusters
            }
        }
