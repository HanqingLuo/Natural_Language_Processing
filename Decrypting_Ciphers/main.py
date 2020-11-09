# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python

# Author: http://lazyprogrammer.me

import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os
import textwrap

### create substitution cipher

# # one will act as the key, other as the value
# # create a list of 26 letters
# letters1 = list(string.ascii_lowercase)
# letters2 = list(string.ascii_lowercase)
# print('letters1 = ', letters1)
# print('letters2 = ', letters2)


# # create dictionay, key from letters1 maps value from letters2
# true_mapping = {}

# # shuffle second set of letters
# random.shuffle(letters2)
# print('random letters2 = ', letters2)

# # populate map
# for k, v in zip(letters1, letters2):
#   true_mapping[k] = v
# print('true_mapping = ', true_mapping)