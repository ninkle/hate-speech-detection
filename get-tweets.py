"""
Uses the Twitter API to extract labelled (racism, sexism, none) tweets from annotated database
sourced from https://github.com/zeerakw/hatespeech.

@InProceedings{waseem-hovy:2016:N16-2,
  author    = {Waseem, Zeerak  and  Hovy, Dirk},
  title     = {Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter},
  booktitle = {Proceedings of the NAACL Student Research Workshop},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  pages     = {88--93},
  url       = {http://www.aclweb.org/anthology/N16-2013}
}

"""


import os
import keys
import re
from twython import Twython
import pandas as pd

# convert csv to dataframe and each column to a list
df = pd.read_csv('NAACL_SRW_2016.csv')
ids = df.ix[:, 0].tolist()
labels = df.ix[:, 1].tolist()

# set up api key/token
t = Twython(app_key=keys.myappkey,
    app_secret=keys.my_app_key,
    oauth_token=keys.my_oauth_token,
    oauth_token_secret=keys.my_oauth_token_secret)

# get tweets, strip all punctuation and write to individual text file labelled by category and tweet id
for i in ids[:1970]:
    f = 'racism%i.txt' % i
    if not os.path.isfile(f):
        try:
            tweet = t.show_status(id=i)['text']
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
            outfp = open(f, "w")
            outfp.write(tweet)
            print('okay')
        except Exception as e:
            print(e)

for i in ids[1971:5350]:
    f = 'sexism%i.txt' % i
    if not os.path.isfile(f):
        try:
            tweet = t.show_status(id=i)['text']
            outfp = open(f, "w")
            outfp.write(tweet)
            print('okay')
        except Exception as e: # Sometimes tweets are deleted, users are blocked, etc OR rate limit exceeded.
            print(e)

for i in ids[5351:16907]:
    f = 'none%i.txt' % i
    if not os.path.isfile(f):
        try:
            tweet = t.show_status(id=i)['text']
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
            outfp = open(f, "w")
            outfp.write(tweet)
            print('okay')
        except Exception as e:  # Sometimes tweets are deleted, users are blocked, etc OR rate limit exceeded.
            print(e)


