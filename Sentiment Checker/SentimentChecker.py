from textblob import TextBlob
from newspaper import Article

url = 'https://www.firstpost.com/explainers/mangaluru-fazil-murder-accused-suhas-shetty-ex-bajranj-dal-killed-13884920.html'
article = Article(url)

article.download()
article.parse()

text = article.text

blob = TextBlob(text)

sentiment = blob.sentiment.polarity
print(sentiment)