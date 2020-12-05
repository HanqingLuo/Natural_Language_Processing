# Very basic article spinner for NLP class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

# Author: http://lazyprogrammer.me

# A very bad article spinner using trigrams.
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import nltk
import random
import numpy as np


'''
Beautiful Soup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
Python library
'''
from bs4 import BeautifulSoup 


# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), "lxml")
positive_reviews = positive_reviews.findAll('review_text')

# print(positive_reviews)
'''
Expected Output:
[
    <review_text> aaaaa </review_text>,
    <review_text> bbbbb </review_text>,
    <review_text> ccccc </review_text>,
    ...
]
'''


# extract trigrams and insert into dictionary
# (w1, w3) is the key, [ w2 ] are the values
'''
Collect all the trigrams
E.g. "I love cats", "I like cats", "I hate cats"
    {("I", "cats") -> ("love", "like", "hate")}
    KEY = (previous word[w1], next word[w3])
    VALUE = possible middle word[w2]
'''
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s) 
    for i in range(len(tokens) - 2): # 每条评论的第一个单词 到 倒数3个单词做trigram dataset
        k = (tokens[i], tokens[i+2]) # w1 = tokens[i], w3 = tokens[i+2]
        if k not in trigrams: # ？？？
            trigrams[k] = [] 
        trigrams[k].append(tokens[i+1]) # Append [w2] to trigrams
# print(trigrams)
'''
Expected output:
('i', 'this'): ['purchased', 'bought', 'bought', 'recomend', 'made', 'picked', 'say', 'bought', 'purchased', 'use', 'bought', 'had', 'bought', 'got', 'got', 'purchased', 'think', 'use', 'ordered', 'matched', 'bought', 'think', 'bought', 'picked', 'picked', 'noticed', 'ordered', 'purchased', 'bought', 'use', 'bought', 'purchased', 'bought', 'thought', 'recommend', 'got', 'bought', 'use', 'use', 'bought', 'choose', 'like', 'bought', 'purchased', 'found', 'got', 'got', 'bought', 'did', 'purchased', 'find', 'did', 'bought', 'purchased', 'bought', 'purchased', 'set', 'love', 'hold', 'bought', 'purchased', 'bought', 'purchased', 'bought', 'found', 'purchased', 'received', 'made', 'have', 'bought', 'bought', 'found', 'find', 'buy', 'bought', 'got', 'found', 'bought', 'found', 'chose', 'think', 'bought', 'purchased', 'bought', 'bought', 'love', 'chose', 'bought', 'bought', 'got', 'put', 'bought', 'used', 'bought', 'bought', 'recommend', 'bought', 'have', 'used', 'purchased', 'give', 'believe', 'recommend', 'bought', 'mention', 'use', 'bought', 'bought', 'bought', 'worked', 'bought', 'bet', 'purchased', 'bought', 'think', 'went', 'pop', 'found', 'have', 'bought', 'use', 'selected', 'purchased', 'bought', 'bought', 'like', 'use', 'bought', 'heard', 'bought', 'use', 'do', 'bought', 'love', 'purchased', 'purchased', 'checked', 'find', 'bought', 'think', 'bought', 'got', 'plugged', 'recommend', 'consider', 'love', 'found', 'bought', 'got', 'think', 'bought', 'reccomend', 'purchased', 'purchased', 'find', 'bought', 'found', 'received', 'did', 'found', 'bought']
...
'''

# turn each array of middle-words into a probability vector
'''
Each middle word needs to be associated with a probilitity
E.g. "I love cats", "I like cats", "I hate cats"
    p(like | I, cats) = 0.5
    p(love | I, cats) = 0.3
    p(hate | I, cats) = 0.2

    {
        ("I", "cats"): {
            "like": 0.5,
            "love": 0.3,
            "hate": 0.2,
        }
    }
'''

# Transfer possible words into possbility vector
trigram_probabilities = {}
for k, words in iteritems(trigrams):
    # create a dictionary of word -> count
    if len(set(words)) > 1:
        # only do this when there are different possibilities for a middle word
        d = {}
        n = 0 
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in iteritems(d): # w = word, c = count
            d[w] = float(c) / n
        trigram_probabilities[k] = d
# print(trigram_probabilities)
'''
expected  outcome
{('i', 'this'): {'purchased': 0.12422360248447205, 'bought': 0.3105590062111801, 'recomend': 0.006211180124223602, 'made': 0.012422360248447204, 'picked': 0.018633540372670808, 'say': 0.006211180124223602, 'use': 0.055900621118012424, ' 
'''

def random_sample(d):
    # choose a random sample from dictionary where values are the probabilities
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original:", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2: # 20% chance of replacement
            k = (tokens[i], tokens[i+2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i+1] = w
    print("Spun:")
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()
