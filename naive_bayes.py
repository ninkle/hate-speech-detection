import os
import time
import glob
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# get_tweets.py left us with a lot of ends of contractions from stripping the punctuation
# mostly because I was figuratively sleeping and tried to clean the text before placing it into a permanent file
# anyways, here are all appropriate replacements for those fragments
contractions_dict = {
    'd': 'had',
    'don': 'do',
    'didn': 'did',
    'll': 'will',
    'm': 'am',
    've': 'have',
    're': 'are',
    's': 'is',
    't': 'not',
    'wouldn': 'would'
    }

def get_stems(text):
    # initialize snowball stemmer from nltk
    snowball = SnowballStemmer("english")

    # initialize count vectorizer from sklearn, creating bag of ngrams
    ngram_vectorizer = CountVectorizer('char_wb', ngram_range=(1, 6))

    # delineate words by empty character " " and converts to list format
    tokenized = (ngram_vectorizer.build_tokenizer()(text))

    # apply snowball stemmer so we can account for grammatical discrepancies
    for i in range(len(tokenized)):
        tokenized[i] = snowball.stem(tokenized[i])

    # convert list back to string and return
    text = (" ".join(tokenized))
    return text

def process(label):
    tweets = []
    twitter_words = ['rt', 'RT', 'dm', 'DM', 'MKR', 'mkr']
    for filename in glob.iglob(os.path.join(label, '*.txt')):
        f = open(filename, "r")
        text = f.read()
        text = text.split()
        text = [i for i in text if i not in twitter_words]
        text = ' '.join(text)
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
        result = pattern.sub(lambda x: contractions_dict[x.group()], text)
        result = get_stems(result)
        tweets.append(result)
    return tweets

if __name__ == '__main__':

    # clean up each text file and append string to list (racist_data, sexism_data, neutral_data, respectively)
    racism_data = process('Racist')
    sexism_data = process('Sexist')
    neutral_data = process('None')

    # create list of labels for each category
    racism_labels = ['racism'] * len(racism_data)
    sexism_labels = ['sexism'] * len(sexism_data)
    neutral_labels = ['neutral'] * len(neutral_data)

    # split into training and testing sets
    # calculate number of tweets constituting 80% of each dataset
    racism_train_size = int(len(racism_data) * 0.8)
    sexism_train_size = int(len(sexism_data) * 0.8)
    neutral_train_size = int(len(neutral_data) * 0.8)

    # split each dataset
    racism_train = racism_data[:racism_train_size]
    sexism_train = sexism_data[:sexism_train_size]
    neutral_train = neutral_data[:neutral_train_size]

    racism_test = racism_data[racism_train_size:]
    sexism_test = sexism_data[sexism_train_size:]
    neutral_test = neutral_data[neutral_train_size:]

    # aggregate all training/test data/labels into respective sets
    train_data = racism_train + sexism_train + neutral_train
    train_labels = racism_labels[:racism_train_size] + sexism_labels[:sexism_train_size] + neutral_labels[:neutral_train_size]

    test_data = racism_test + sexism_test + neutral_test
    test_labels = racism_labels[racism_train_size:] + sexism_labels[sexism_train_size:] + neutral_labels[neutral_train_size:]

    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 decode_error='ignore')

    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # classification with Multinomial Naive Bayes
    mnb = MultinomialNB()
    t0 = time.time()
    mnb.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_mnb = mnb.predict(test_vectors)
    t2 = time.time()
    time_mnb_train = t1 - t0
    time_mnb_predict = t2 - t1

    # classification with Bernoulli Naive Bayes
    bnb = BernoulliNB()
    t0 = time.time()
    bnb.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_bnb = bnb.predict(test_vectors)
    t2 = time.time()
    time_bnb_train = t1 - t0
    time_bnb_predict = t2 - t1

    # print results
    print("Results for MultinomialNB()")
    print("Training time: %fs; Prediction time: %fs" % (time_mnb_train, time_mnb_predict))
    print(classification_report(test_labels, prediction_mnb))

    print("Results for BernoulliNB()")
    print("Training time: %fs; Prediction time: %fs" % (time_bnb_train, time_bnb_predict))
    print(classification_report(test_labels, prediction_bnb))





