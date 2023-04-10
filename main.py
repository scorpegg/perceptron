import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

with open('негативные.txt', 'r', encoding='utf-8') as f:
    neg_reviews = f.readlines()

with open('положительные.txt', 'r', encoding='utf-8') as f:
    pos_reviews = f.readlines()

neg_reviews = [review.strip() for review in neg_reviews if review.strip() and review[0].isdigit()]
pos_reviews = [review.strip() for review in pos_reviews if review.strip() and review[0].isdigit()]

stop_words = set(stopwords.words('russian'))
punctuations = set(string.punctuation)

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in punctuations]
    return tokens

neg_reviews = [clean_text(review) for review in neg_reviews]
pos_reviews = [clean_text(review) for review in pos_reviews]

word_dict = {}
for review in neg_reviews + pos_reviews:
    for word in review:
        if word not in word_dict:
            word_dict[word] = len(word_dict)

def create_vector(review):
    vector = [0] * len(word_dict)
    for word in review:
        if word in word_dict:
            vector[word_dict[word]] += 1
    return vector

neg_vectors = [create_vector(review) for review in neg_reviews]
pos_vectors = [create_vector(review) for review in pos_reviews]

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

X = neg_vectors + pos_vectors
y = [0] * len(neg_vectors) + [1] * len(pos_vectors)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Perceptron()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

def predict_sentiment(text):
    tokens = clean_text(text)
    vector = create_vector(tokens)
    sentiment = clf.predict([vector])[0]
    if sentiment == 0:
        return 'Negative'
    else:
        return 'Positive'