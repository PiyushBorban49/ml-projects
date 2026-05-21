import tkinter as tk
import nltk
from newspaper import Article
from nltk.downloader import TKINTER
from textblob import TextBlob
nltk.download('punkt_tab')

def summarize():

    url = Url.get('1.0',"end").strip()

    # url = "https://smartone.ai/blog/best-machine-learning-datasets-for-chatbot-training/"
    article = Article(url)

    article.download()
    article.parse()
    article.nlp()

    title.config(state="normal")
    Author.config(state="normal")
    Publication_data.config(state="normal")
    Summary.config(state="normal")
    Polarity.config(state="normal")

    title.delete('1.0',"end")
    title.insert('1.0',article.title)

    Author.delete('1.0', "end")
    Author.insert('1.0', article.authors)

    Publication_data.delete('1.0', "end")
    Publication_data.insert('1.0', article.publish_date)

    Summary.delete('1.0', "end")
    Summary.insert('1.0', article.summary)

    Polarity.delete('1.0', "end")
    Polarity.insert('1.0', "Positive" if TextBlob(article.text).polarity > 0 else "Negative" if TextBlob(article.text).polarity < 0 else "Neutral")


    title.config(state="disabled")
    Author.config(state="disabled")
    Publication_data.config(state="disabled")
    Summary.config(state="disabled")
    Polarity.config(state="disabled")

    # article_summary = article.summary
    # print(article_summary)
    #
    # analysis = TextBlob(article.text)
    # print(analysis.polarity)

root = tk.Tk()

root.title("Article Summarizer")
root.geometry("1200x600")

tlabel = tk.Label(root,text="Title")
tlabel.pack()

title = tk.Text(root,height=1,width=140)
title.config(state="disabled",bg='#dddddd')
title.pack()

tAuthor = tk.Label(root,text="Author")
tAuthor.pack()

Author = tk.Text(root,height=1,width=140)
Author.config(state="disabled",bg='#dddddd')
Author.pack()

tPublication_date = tk.Label(root,text="Publication Date")
tPublication_date.pack()

Publication_data = tk.Text(root,height=1,width=140)
Publication_data.config(state="disabled",bg='#dddddd')
Publication_data.pack()

tSummary = tk.Label(root,text="Summary")
tSummary.pack()

Summary = tk.Text(root,height=20,width=140)
Summary.config(state="disabled",bg='#dddddd')
Summary.pack()

tPolarity = tk.Label(root,text="Sentiment Analysis")
tPolarity.pack()

Polarity = tk.Text(root,height=1,width=140)
Polarity.config(state="disabled",bg='#dddddd')
Polarity.pack()

tUrl = tk.Label(root,text="URL")
tUrl.pack()

Url = tk.Text(root,height=1,width=140)
Url.pack()

btn = tk.Button(root,height=1,width=10,text="Summarize",command=summarize)
btn.config(state="normal",bg='#dddddd')
btn.pack()






root.mainloop()







