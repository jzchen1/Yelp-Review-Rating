# -*- coding: utf-8 -*-
import csv
import nltk
import numpy as np
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.sentiment.util import mark_negation
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas
from tkinter import *
from tkinter import scrolledtext
from PIL import Image, ImageTk
import random

#Preprocessor - Creating Dataset

stop_words = set(stopwords.words('english'))

#Requires the review.csv file to run

f = open('review.csv', "r", encoding = "utf-8")
read = csv.reader(f)
counter = 0
text_reviews = []
stars = []
for row in read:
    counter += 1
    if(counter < 100000): 
        if row[5] != '3':
            stars.append(row[5])
            text_reviews.append(row[6])
    else:
        f.close()
        text_reviews = text_reviews[1:]
        stars = stars[1:]
        break

# Preprocessor - Creating "Balanced Subset"

n = Counter(stars)
max_ = n.most_common()[-1][1]
n_added = {class_: 0 for class_ in n.keys()}
new_ys = []
new_xs = []
for i, y in enumerate(stars):
    if n_added[y] < max_:
        new_ys.append(y)
        new_xs.append(text_reviews[i])
        n_added[y] += 1
        
stars = new_ys
text_reviews = new_xs

#Taking User Input for 10 random Reviews
random_nums = list()
for x in range(len(text_reviews)):
    random_nums.append(x)

ten_stars = list()
ten_text = list()
user_prediction = list()

for x in range(10):
    random_review = random.choice(random_nums)
    print("\n")
    print(text_reviews[random_review])
    var = "0"
    
    while var not in ["1", "2", "4", "5"]:
        var = input("Please the rating you think this review recieved out of [1, 2, 4, 5] stars: ")
    
    user_prediction.append(var)

    ten_stars.append(stars.pop(random_review))
    ten_text.append(text_reviews.pop(random_review))
    random_nums.pop(-1)
    
'''
#Preprocessor - Lemmatizer

xs_dupe = []
l = WordNetLemmatizer()
for x in text_reviews:
    review = str()
    for w in word_tokenize(x):
        current = str(l.lemmatize(w, pos = 'n'))
        for t in [l.lemmatize(w, pos = 'v'), l.lemmatize(w, pos = 'a'), l.lemmatize(w, pos = 'r')]:
            if len(t) < len(current):
                current = t
        if '\'' in current or '!' in current or '.' in current or ',' in current:
            review = review + current
        else:
            review = review + ' ' + current
    xs_dupe.append(review)
text_reviews = xs_dupe
'''

print("Estimated Loading Time 1-2minutes ......")

#Vectorizer - n-grams, removing stopwords, negation detection

vectorizer = CountVectorizer(ngram_range=(1,3), stop_words = 'english', tokenizer=lambda text: mark_negation(word_tokenize(text)))
vectors = vectorizer.fit_transform(text_reviews)
user_tested = vectorizer.transform(ten_text)
stars = np.array(stars)

#Training the Multinomial Naive Bayes Classifier and testing it against the same reviews the user was asked. 
nb = MultinomialNB()
nb.fit(vectors, stars)
y_pred = nb.predict(user_tested)
classifier_accuracy_list = list()


classifier_accuracy_list.append(metrics.accuracy_score(ten_stars, y_pred))
classifier_accuracy_list = np.array(classifier_accuracy_list)

print("\n")
print("AI's Classification Accuracy = ", classifier_accuracy_list.mean())
print("User's Classification Accuary = ", metrics.accuracy_score(ten_stars, user_prediction))

