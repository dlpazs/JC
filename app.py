from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap 
from textblob import TextBlob,Word 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
import csv
from flask_bootstrap import Bootstrap 
import random 
import time
from flask import jsonify
import json
import requests
import matplotlib.pyplot as plt

#New Variables
# https://www.rivs.com/blog/personality-traits/
# Which jobs suit which personalities
#https://www.myersbriggs.org/my-mbti-personality-type/mbti-basics/home.htm?bhcp=1
#http://uk.businessinsider.com/best-jobs-for-every-personality-2014-9

#New research breaking personality 5 into smaller traits
#https://www.mindtools.com/pages/article/newCDV_22.htm
#http://career.iresearchnet.com/career-assessment/big-five-factors-of-personality/
#https://www.independent.co.uk/student/student-life/Studies/new-study-finds-link-between-big-five-personality-traits-and-which-subject-students-study-at-a6846996.html

#NLTK
import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords, state_union
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/form', methods=['POST'])
def form():
	return render_template('form.html')

@app.route('/result', methods=['POST', 'GET'])
def predict():
	if request.method == 'POST':

		#form inputs from user
		Open = request.form['Openn']
		Consc = request.form['Consc']
		Extra = request.form['Extra']
		Agree = request.form['Agree']
		Emoti = request.form['Emoti']
		Senso = request.form['Sensors']
		Decis = request.form['Decision']
		Organ = request.form['Organisation']

		#SVM Classifier
		df = pd.read_csv('Sac.csv')
		x = df.drop(['job_title', 'description'], 1)#drop the Y output
		y = df['job_title'] # Y output label
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)#Train test split
		#parameters = {'kernel':('linear'), 'C':[1,10,100,1000], 'gamma':[1,0.1,0.001,0.0001,1e-3, 1e-4]} 
		#GridSearchCV(SVC(probability=True), parameters) 
		clf = svm.SVC()#gridsearchCV
		clf.fit(x_train, y_train)#Fit the classifier

		accuracy = clf.score(x_test, y_test)
		example = [[Open,Consc,Extra,Agree,Emoti,Senso,Decis,Organ]]
		prediction = []
		prediction = clf.predict(example)

	return render_template('result.html', data_obj =  prediction, data_obj1 = example)



#@app.route('/chat', methods=['POST'])
#def ml():
#	start = time.time()
	#Tokenizer
#	rawtext = request.form['rawtext']
#	custom_sent_tokenizer = PunktSentenceTokenizer(rawtext)
#	tokenized = custom_sent_tokenizer.tokenize(rawtext)

		#NLP Stuff
#	blob = TextBlob(rawtext)
#	received_text2 = blob

#	blob_sent = blob.sentiment.polarity
#	number_of_tokens = len(list(blob.words))

#	jsonify(blob_sent)

#	unhappy = ''

#	if (blob_sent <= -0.4 and blob_sent > -0.7):
#		unhappy = 'The sentiment is bad'
#	elif (blob_sent <= -0.7 ):
#		unhappy = 'The sentiment is very bad! ' 
#	elif (blob_sent > -0.4 and blob_sent < 0.4 ):
#		unhappy = 'The sentiment is not bad or good'
#	elif (blob_sent >= 0.4 and blob_sent < 0.7):
#		unhappy = 'The sentiment is good'
#	elif (blob_sent >= 0.7 ):
#		unhappy = 'The sentiment is very good!'

		# Extracting Main Points
#	nouns = list()
#	len_of_words = len(nouns)
#	for word, tag in blob.tags:
#		if tag == 'NN':
#		    nouns.append(word.lemmatize())
#		    len_of_words = len(nouns)
#		    rand_words = random.sample(nouns,len(nouns))
#		    final_word = list()
#		    for item in rand_words:
#		        word = Word(item).pluralize()
#		        final_word.append(word)
#		        summary = final_word
#		        end = time.time()
#		        final_time = end-start


	
#	return render_template('chat.html', received_text = received_text2, sent_score = blob_sent, blob_sent = unhappy, number_of_tokens=number_of_tokens, len_of_words = len_of_words)

#chatbot imports
from chatterbot import ChatBot 
from chatterbot.trainers import ListTrainer
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

@app.route('/chat-bot', methods=['POST'])
def chat_bot():

	df = pd.read_csv('./new.csv', encoding = "ISO-8859-1")
	count_vect = CountVectorizer()
	tfidf_transformer = TfidfTransformer()

	text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),])

	x2 = df['TEXT']

	yEXT = df['cEXT']  
	yNEU = df['cNEU']
	yAGR = df['cAGR']
	yCON = df['cCON']
	yOPN = df['cOPN']

	#
	X_train, X_test, y_train, y_test = train_test_split(x2, yEXT, random_state = 0)
	X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, yNEU, random_state = 0)
	X_trainAGR, X_testAGR, y_trainAGR, y_testAGR = train_test_split(x2, yAGR, random_state = 0)
	X_trainCON, X_testCON, y_trainCON, y_testCON = train_test_split(x2, yCON, random_state = 0)
	X_trainOPN, X_testOPN, y_trainOPN, y_testOPN = train_test_split(x2, yOPN, random_state = 0)

	# One-hot encoding (CountVectorizing)
	#Transforming words into victors by count occurrence of each word in a document
	X_train_counts = count_vect.fit_transform(X_train)
	X_train_counts2 = count_vect.fit_transform(X_train2)
	X_train_countsAGR = count_vect.fit_transform(X_trainAGR)
	X_train_countsCON = count_vect.fit_transform(X_trainCON)
	X_train_countsOPN = count_vect.fit_transform(X_trainOPN)

	# TfidfTransformer
	#the goal of tf-idf instead of raw frequencies of occurence of a token in a given document is to scale down the impact
	#of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features
	#that occur in a small fraction of the training corpus
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X_train_tfidf2 = tfidf_transformer.fit_transform(X_train_counts2)
	X_train_tfidfAGR = tfidf_transformer.fit_transform(X_train_countsAGR)
	X_train_tfidfCON = tfidf_transformer.fit_transform(X_train_countsCON)
	X_train_tfidfOPN = tfidf_transformer.fit_transform(X_train_countsOPN)
	#
	clf = MultinomialNB().fit(X_train_tfidf, y_train)
	clf2 = MultinomialNB().fit(X_train_tfidf2, y_train2)
	clfAGR = MultinomialNB().fit(X_train_tfidfAGR, y_trainAGR)
	clfCON = MultinomialNB().fit(X_train_tfidfCON, y_trainCON)
	clfOPN = MultinomialNB().fit(X_train_tfidfOPN, y_trainOPN)
	#

	chat_text = request.form['rawtext']
	bot = ChatBot('Bot')
	bot.set_trainer(ListTrainer)

	#ex = count_vect.fit_transform([chat_text])
	example = count_vect.transform([chat_text])

	predicted = clf.predict(example)
	predicted2 = clf2.predict(example)
	predictedAGR = clfAGR.predict(example)
	predictedCON = clfCON.predict(example)
	predictedOPN = clfOPN.predict(example)

	chat = TextBlob(chat_text)
	chat_Sentiment = chat.sentiment.polarity
	reply = bot.get_response(chat_text)
#	while True:
#		message = request.form['rawtext']
		
	#	if message.strip() != 'Bye':
	#		reply = bot.get_response(message)
#		if message.strip() == 'Bye':
	#		break

	return render_template('chat-bot.html',received_text = chat_text, bot_response = reply, chat_sent = chat_Sentiment, pers_response = predictedAGR, EXT = predicted, NEU = predicted2,
		CON = predictedCON, OPN = predictedOPN)


if __name__ == '__main__':
	#app.run(debug=True)
	app.run(debug=True)




#Part of speech tagging
# Labelling words in a sentence as nouns, adjective, verbs etc. And it also labels by tense. 
#PunktSetenceTokenizer
#we can begin to derive meaning, but there is still some work to do.
# The next topic that we're going to cover is chunking, 
#which is where we group words, based on their parts of speech, into hopefully meaningful groups.
#main goals of chunking is to group into what are known as "noun phrases." 
#These are phrases of one or more words that contain a noun, maybe some descriptive words, 
#maybe a verb, and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.
#A very similar operation to stemming is called lemmatizing. The major difference 
#between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
#
#from gensim.models import word2vec
#corpus = [
 #         'Text of the first document.',
  #        'Text of the second document made longer.',
   #       'Number three.',
    #      'This is number four.',
#]
# we need to pass splitted sentences to the model
#tokenized_sentences = [sentence.split() for sentence in corpus]
#model = word2vec.Word2Vec(tokenized_sentences, min_count=1)