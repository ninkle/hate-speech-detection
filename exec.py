from sklearn.externals import joblib
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re

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

def process(text):
    twitter_words = ['rt', 'RT', 'dm', 'DM', 'MKR', 'mkr']
    text = text.split()
    text = [i for i in text if i not in twitter_words]
    text = ' '.join(text)
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    result = pattern.sub(lambda x: contractions_dict[x.group()], text)
    result = [get_stems(result)]
    return result

if __name__ == '__main__':
    clf = joblib.load('classifier.pkl')
    vec = joblib.load('vectorizer.pkl')
    cond = True
    while cond == True:
        x = input("Enter some example text:")
        if x == "q":
            cond = False
        else:
            x = process(x)
            print(x)
            x_test = vec.transform(x)
            print(clf.predict(x_test))