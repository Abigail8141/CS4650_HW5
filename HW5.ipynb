import en_core_web_lg
import spacy
import pickle
import numpy as np
from newsapi import NewsApiClient
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import nltk
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key='122ffe874e59412d8f5649f679292dd9')

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-09-23', to='2021-10-21', sort_by='relevancy')

filename = 'articlesCOVID.pckl'
pickle.dump(temp, open(filename, 'wb'))

filename = 'articlesCOVID.pckl'
loaded_model = pickle.load(open(filename, 'rb'))

filepath = '/content/articlesCOVID.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

dados = []
for i, article in enumerate(temp):
    for x in temp['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        date = x['publishedAt']
        dados.append({'title':title, 'date':date, 'desc':description, 'content':content})
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

def get_keywords_eng(token):
  result = []
  punctuation = string.punctuation
  stop_words = stopwords.words("english")
  for i in token:
    if (i in stop_words):
      continue
    else:
      result.append(i)
  return result

tokenizer = RegexpTokenizer(r'\w +')
results = []
for content in df.content.values:
    content = tokenizer.tokenize(content)
    results.append([x[0] for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

text = str(results)
wordcloud = WordCloud(max_font_size=30, max_words=150, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

text = str(results)
wordcloud = WordCloud(max_font_size=30, max_words=150, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
