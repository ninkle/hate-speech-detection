import os
import time
import glob
import re
import nltk
from nltk.corpus import names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
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
    ngram_vectorizer = CountVectorizer('char_wb', ngram_range=(2, 8))

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
    female_names = names.words('female.txt')
    for filename in glob.iglob(os.path.join(label, '*.txt')):
        f = open(filename, "r")
        text = f.read()
        text = text.split()
        text = [i for i in text if i not in twitter_words]
        for i in range(len(text)):
            if text[i] in female_names:
                text[i] = 'woman'
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

    # use all data -- comment/uncomment to use/not use full data
    racism_train_size = len(racism_data)
    sexism_train_size = len(sexism_data)
    neutral_train_size = len(neutral_data)

    # comment/uncomment to use/not use split data
    # split into training and testing sets
    # calculate number of tweets constituting 80% of each dataset
    '''racism_train_size = int(len(racism_data) * 0.8)
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
    test_labels = racism_labels[racism_train_size:] + sexism_labels[sexism_train_size:] + neutral_labels[neutral_train_size:]'''

    train_data = racism_data + sexism_data + neutral_data
    train_labels = racism_labels + sexism_labels + neutral_labels


    vectorizer = CountVectorizer()

    train_vectors = vectorizer.fit_transform(train_data)
    # test_vectors = vectorizer.transform(test_data)


    # classification with Linear SVC
    lin = svm.LinearSVC()
    t0 = time.time()
    lin.fit(train_vectors, train_labels)
    t1 = time.time()
    # prediction_lin = lin.predict(test_vectors)
    t2 = time.time()
    time_lin_train = t1 - t0
    time_lin_predict = t2 - t1

    # print results
    # precision = number of correct positive results divided by the number of all positive results
    # recall = number of correct positive results divided by the number of positive results that
    # should have been returned
    # f1-score = weighted average of the precision and recall

    # classification report
    '''print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_lin_train, time_lin_predict))
    print(classification_report(test_labels, prediction_lin))'''

    # Save the model as a pickle in a file
    saved_vectorizer = joblib.dump(vectorizer, 'vectorizer.pkl')
    saved_classifier = joblib.dump(lin, 'classifier.pkl')
