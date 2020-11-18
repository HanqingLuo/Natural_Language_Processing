

from __future__ import print_function, division
# from future.utils import iteritems
from builtins import range
import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from matplotlib import pyplot as plt



'''
	All ML interfaces are the same. We can plug in any other classifier, such as AdaBoost, without modifying the data or any surrounding code 

		model = Model()
		model.fit(X,Y)
		model.predict(X)
'''


'''
1. Load data
'''
# load our data
# encoding: cvs countains invalid characters, text 
df = pd.read_csv('C:/Users/Hanson/Documents/GitHub/Natural_Language_Processing/Spam_Detector/SD_raw_data/spam.csv')
#print(df)

'''
2. Data Cleanup
'''
# drop unnecessary columns 
df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)
#print(df)

# rename colums to something better
df.columns = ['labels', 'data']
# print(df)

'''
3. Data Processing
'''
# create binary labels 
df['b_labels'] = df['labels'].map({'ham': 0,'spam': 1}) # add one column 'b_labels' 
#print(df)
Y = df['b_labels'].to_numpy()
print(Y)
# Expect print(Y):
# [1 rows x 5572 columns] binary value 


# X = input feature for every sample
count_vectorizer = CountVectorizer(decode_error = 'ignore')
X = count_vectorizer.fit_transform(df['data'])
print(X)

'''
4. Data split
'''
# split up the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y)
# train_test_split(X, y, train_size=0.*,test_size=0.*, random_state=*)
# train_size: This parameter sets the size of the training dataset. There are three options: None, which is the default, Int, which requires the exact number of samples, and float, which ranges from 0.1 to 1.0.
# test_size: This parameter specifies the size of the testing dataset. The default state suits the training size. It will be set to 0.25 if the training size is set to default.

# create the model, train it, print scores
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain)) 
print("test score:", model.score(Xtest, Ytest)) 

# Visualize the data
def Visualize(label):
	words = ''
	for msg in df[df['labels'] == label]['data']:
		msg = msg.lower()
		words += msg + ' '
	wordcloud = WordCloud(width = 600, height = 400).generate(words)
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()

# Visualize('spam')
# Visualize('ham')

# see what we're getting wrong 
df['predictions'] = model.predict(X)

# things that should be spam
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in sneaky_spam:
	print(msg)

# things that should NOT be spam
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg in not_actually_spam:
	print(msg)











