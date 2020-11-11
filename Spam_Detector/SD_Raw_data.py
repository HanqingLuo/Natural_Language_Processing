from sklearn.naive_bayes import MultinomialNB
import pandas as pd 
import numpy as np


'''


	All ML interfaces are the same. We can plug in any other classifier, such as AdaBoost, without modifying the data or any surrounding code 

		model = Model()
		model.fit(X,Y)
		model.predict(X)

	

'''



# load our data
data = pd.read_csv('spam.csv').to_numpy()
np.random.shuffle(data)

# Preprocess the data





X = data[:,:48] # first 48 cols is our data input
Y = data[:,-1] # all the rows, last cols

Xtrain = X[:-100,] # first 100 rows
Ytrain = Y[:-100,]
Xtest = X[-100:,] # last 100 rows
Ytest = Y[-100:,] # last 100 rows

model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest)) # 80+% accuracy

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print("Classification rate for AdaBoost:", model.score(Xtest, Ytest)) 










