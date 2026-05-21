import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorboard.compat.tensorflow_stub.tensor_shape import vector

nltk.download('stopwords')

data = pd.read_csv('spam_ham_dataset.csv')
data.drop('label',axis=1)
data['text'] = data['text'].apply(lambda x : x.replace('\r\n',' '))

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
y = data.label_num

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train,Y_train)

# In this way you can check your email
# email = data.text.values[10].lower()
# email = email.translate(str.maketrans('','',string.punctuation)).split()
# email = [stemmer.stem(word) for word in text if word not in stopwords_set]
# email = ' '.join(email)
# X_email = vector.transform([email])
# print(clf.predict(X_email))

print(clf.score(X_test,Y_test))