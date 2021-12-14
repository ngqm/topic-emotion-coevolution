"""
Authors: Quang Minh Nguyen and Maida Aizaz
Save the topics of a given month into a json file
"""

import json
import pandas as pd
from nrclex import NRCLex

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from time import time

t0 = time()

# -------------------------------------------------------------------
# CONSTANTS
MONTH = '2017-6'


# ----------------------------------------------------------------- #
# IMPORT DATA FROM FILES
data = pd.read_csv(f"data/data-{MONTH}.csv").content 


# -------------------------------------------------------------------
# SAVE TOPICS

# bag of words representation
tf_vectorizer = CountVectorizer(
    max_df=.7,  # words occuring in >70% of documents are excluded
    min_df=5,   # words occuring in <5 document are excluded
    stop_words="english")

# latent dirchlet allocation for topic discovery
lda = LatentDirichletAllocation(
    learning_method="online",
    learning_offset=50.0,
    max_iter=5,
    random_state=0)

# create document-term matrix
tf = tf_vectorizer.fit_transform(data.values.astype('str'))
# generate latent dirichlet allocation model
lda.fit(tf)
# get words in topics
tf_feature_names = tf_vectorizer.get_feature_names()

topics = []
emotions = []
for topic in lda.components_:
  # save topics
  counts = {word: int(count) for word, count in zip(tf_feature_names, topic) if int(count)}
  topics.append(counts)
  # save emotions
  string = ''.join([(word + ' ')*count for word, count in counts.items()])
  emotion = NRCLex(string).raw_emotion_scores
  emotions.append(emotion)

# dump topics to json file
with open(f'topics/topics-{MONTH}.json', 'w') as file:
  json.dump(topics, file)
# dump emotions to json file
with open(f'emotions/emotions-{MONTH}.json', 'w') as file:
  json.dump(emotions, file)

print('Done! That took us {}s'.format(time()-t0))