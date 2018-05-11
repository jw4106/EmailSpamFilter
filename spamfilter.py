import os
import re
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
from stop_list import *

class Email:
	eid = 0
	features = []
	body = []

	def __init__(self, eid, body):
		self.eid = eid
		self.body = body

allwords = {}
allemails_train = {}
allemails_test = {}
totaltrainfiles = 0
totaltestfiles = 0

enron2path = 'enron2/spam/'
spampath = 'spam_test/'
trainpath = 'train-mails/'

#TRAIN: parse each spam email and create a common word bank
for i, email in enumerate(sorted(os.listdir(spampath))):
	with open(spampath+email, encoding='utf8', errors='ignore') as file:
		totaltrainfiles+=1
		data = file.read()
		data = data.split('\n', 1)
		body = data[1].replace('\n', ' ')
		body = re.sub(r'[^\w]', ' ', body)
		words = body.split()
		tempemail = Email(i, words)
		allemails_train[tempemail.eid] = tempemail
		for word in words:
			if word.isalpha() and len(word) != 1:
				if word in allwords:
					allwords[word]+=1
				else:
					allwords[word] = 1

#TEST: parse each spam email and create a common word bank
for i, email in enumerate(sorted(os.listdir('test-mails/'))):
	with open('test-mails/'+email, encoding='utf8', errors='ignore') as file:
		totaltestfiles+=1
		data = file.read()
		data = data.split('\n', 1)
		body = data[1].replace('\n', ' ')
		body = re.sub(r'[^\w]', ' ', body)
		words = body.split()
		tempemail = Email(i, words)
		allemails_test[tempemail.eid] = tempemail				

#TRAIN: set up common words size:40
for stopword in closed_class_stop_words:
	allwords.pop(stopword, None)

commonWords = sorted(allwords, key=allwords.get, reverse=True)[:3000]


#TRAIN: set up features for each
for key,value in allemails_train.items():
	singlefeature = np.zeros(3000)  
	for index, word in enumerate(commonWords):
		for bodyword in value.body:
			if bodyword == word:
				singlefeature[index]+=1
	allemails_train[key].feature = singlefeature

#TEST: set up features for each
for key,value in allemails_test.items():
	singlefeature = np.zeros(3000)  
	for index, word in enumerate(commonWords):
		for bodyword in value.body:
			if bodyword == word:
				singlefeature[index]+=1
	allemails_test[key].feature = singlefeature	
	 
#TRAIN: feature matrix
featurematrix = allemails_train[0].feature
for key, value in allemails_train.items():
	if key > 0:
		featurematrix = np.row_stack((featurematrix, value.feature))

#TEST: feature matrix
featurematrix_test = allemails_test[0].feature
for key, value in allemails_test.items():
	if key > 0:
		featurematrix_test = np.row_stack((featurematrix_test, value.feature))		

print(featurematrix)
print(commonWords)		
print(totaltrainfiles)
print(totaltestfiles)



train_labels = np.zeros(totaltrainfiles)
train_labels[351:701] = 1
featurematrix_train = featurematrix
naives = MultinomialNB()
svm = LinearSVC()
naives.fit(featurematrix_train, train_labels)
svm.fit(featurematrix_train, train_labels)

test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = naives.predict(featurematrix_test)
result2 = svm.predict(featurematrix_test)
print("Naive Bayes: ")
print(confusion_matrix(test_labels,result1))
print("Support Vector Machines: ")
print(confusion_matrix(test_labels,result2))

#make_dictionary(train_dir): 
#makes a dictionary with word count
#takes in a directory to find files 
#returns the dictionary

#extract_features(mail_dir)
#opens all the files and creates a 2d matrix 
#returns feature matrix 


