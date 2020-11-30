'''
NLTK Tutorial
https://riptutorial.com/nltk/topic/10028/pos-tagging
'''



import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download()
 

# POS_Tagging
tag1 = nltk.pos_tag('Hanson is awesome!'.split())
print(tag1)

# Expect output: [('Hanson', 'NNP'), ('is', 'VBZ'), ('awesome!', 'JJ')]

'''
Stemming
* Reduce words to a "base" from
e.g. dogs/dog, jump/jumping
'''
from nltk.stem import PorterStemmer
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

# Stem a word
ps = PorterStemmer()
for w in example_words:
    print(ps.stem(w))

# Stem a sentence after tokenizing it
new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
word_tokens = word_tokenize(new_text)
for w in word_tokens:
    print(ps.stem(w))   # Passing word tokens into stem method of Porter Stemmer


'''
NER(Named Entity Recognition)
Examples of entities:
* Albert Einstein -> Person
* Apple -> Organization
'''
s = "Hanson is the most awesome guy, who was born on November 15, 1993, in this planet"
s2 = "Steve Jobs was the CEO of Apple Corp"
tag1 = nltk.pos_tag(s.split())
tag2 = nltk.pos_tag(s2.split())
# print(tag2)

chunk1 = nltk.ne_chunk(tag1)
chunk2 = nltk.ne_chunk(tag2)
print(chunk1)
nltk.ne_chunk(tag1).draw()
nltk.ne_chunk(tag2).draw()