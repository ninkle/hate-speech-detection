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

import sys
import string
import simplejson
import os
import re
from twython import Twython
import pandas as pd
from string import punctuation

# CONVERT CSV TO DATAFRAME, CONVERT COLUMNS TO LISTS
df = pd.read_csv('NAACL_SRW_2016.csv')
ids = df.ix[:, 0].tolist()
labels = df.ix[:, 1].tolist()

# FOR OAUTH AUTHENTICATION -- NEEDED TO ACCESS THE TWITTER API
t = Twython(app_key='1wBxhe4DIFOQg2tU57JBWukSZ',
    app_secret='F3ROiVVS40IWYYFCKW6glTknxEhwoP796Uz0vXrh9c8Hsu829e',
    oauth_token='595554298-oL5VKhdLOAjfucT2YrwPb5bAfHjRSqq52Ki7uKMF',
    oauth_token_secret='ZlfHUFgrLYqDNuF6KI3iiQt9RRJX8TDtEn8SOq7j1zOJ6')

# GET TWEETS AND STRIP USERNAMES, HASHTAGS, AND PUNCTUATION
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
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
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


