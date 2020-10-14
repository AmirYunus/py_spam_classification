import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
df.columns = ['labels', 'data']

df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
target_data = df['b_labels'].values

# tfidf = TfidfVectorizer(decode_error='ignore')
# input_data = tfidf.fit_transform(df['data'])

count_vectorizer = CountVectorizer(decode_error='ignore')
input_data = count_vectorizer.fit_transform(df['data'])

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, test_size = 0.33)

model = MultinomialNB()
model.fit(train_input, train_target)
print(f"train score: {model.score(train_input, train_target)}")
print(f"test score: {model.score(test_input, test_target)}")

def visualise(label):
    words = ''

    for message in df[df['labels'] == label]['data']:
        message = message.lower()
        words += message + ' '
    
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

"""
Uncomment the lines below to visualise the wordcloud or view messages of wrongly predicted messages.
"""

# visualise('spam')
# visualise('ham')

df['predictions'] = model.predict(input_data)
sneaky_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
# for message in sneaky_spam:
#     print(message)

not_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
# for message in not_spam:
#     print(message)