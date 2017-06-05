import tensorflow as tf
import numpy as np
import math
import glob
import os
import re
from nltk.corpus import stopwords
from collections import Counter

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
    'wouldn': 'would',
    'notsexist': 'not sexist'
    }

stops = set(stopwords.words('english'))

# parse out twitter-specific words and stitch up the hanging contractions
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
        tweets.append(result)
    return tweets

def build_vocabulary(data):
    vocab = Counter()
    for tweet in data:
        tweet = tweet.split()
        for word in tweet:
            if word not in stops:
                vocab[word] += 1
    return vocab

if __name__ == '__main__':

    # process text and build our vocabulary
    sexism_data = process('Sexist')
    vocabulary = build_vocabulary(sexism_data)
    vocabulary_size = len(vocabulary)

    # model
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):

        train_dataset = tf.placeholder(tf.int32, shape=[len(sexism_data)])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # define parameters
        embedding_size = len(sexism_data)
        skip_window = 1
        num_skips = 2

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                          stddev=1.0 / math.sqrt(embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # model
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        print("Embed size: %s" % embed.get_shape().as_list())
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                               train_labels, num_sampled, vocabulary_size))

        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)



