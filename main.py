from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("data/data.csv")


data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})


data = data[["tweet", "labels"]]


import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)


x = np.array(data["tweet"])
y = np.array(data["labels"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
decision_tree.score(X_test,y_test)

def hate_speech_detection():

    entered_text = input("Enter any Tweet: ")
    if len(entered_text) < 1:
        print(' ')
    else:
        sample = entered_text
        data = vectorizer.transform([sample]).toarray()
        a = decision_tree.predict(data)
        print(a)
hate_speech_detection()