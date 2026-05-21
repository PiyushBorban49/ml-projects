import numpy as np
import pandas as pd
import string

from nltk import download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

download('stopwords')

data = pd.read_csv('spam.csv')

null_Columns = ['Unnamed2','Unnamed3','Unnamed4']
for i in null_Columns:
    data = data.drop(i , axis=1)

data['label'] = data.label.apply(lambda x : 1 if x == 'spam' else 0)
data['text'] = data.text.apply(lambda x : x.replace('\n\r',' '))

stemmer = PorterStemmer()
corpus = []

stopwords_set = set(stopwords.words(['english']))

for i in range(len(data)):
    text = data['text'].iloc[i].lower()
    text = text.translate(str.maketrans('','',string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

vector = CountVectorizer()
X = vector.fit_transform(corpus).toarray()
y = data.label

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))