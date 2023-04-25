"""
Author: Quang Minh Nguyen and Maida Aizaz
Clean article contents for a given month in a year,
in order:
- Delete links
- Delete non-alphabetical and non-space characters
- Lemmatise
"""

import pandas as pd
import re  # for regex
import sys

# to repress pandas warnings
import warnings
warnings.filterwarnings("ignore")

# for lemmatisation
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# for timing 
from time import time


# ----------------------------------------------------------------- #
# CONSTANTS
MONTH     = sys.argv[1] # month of data to clean
YEAR      = sys.argv[2] # year of data to clean


# ----------------------------------------------------------------- #
# EXTRACT DATA FROM FILES

print("Step 1: Loading data...")
t0 = time()

files = ["data/articles1.csv", 
        "data/articles2.csv",
        "data/articles3.csv"]
data_list = [pd.read_csv(file, index_col=None, header=0) 
    for file in files]
data = pd.concat(data_list, axis=0, ignore_index=True)
# remove trash columns and url
data = data.drop(data.columns[[0, 1, 8]], axis=1)
data = data[(data.year == YEAR) & (data.month == MONTH)]
data = data.reset_index(drop=True)
N_SAMPLES = len(data)
print("Total number of articles: {}".format(N_SAMPLES))

print("Done in %0.3fs." % (time() - t0))


# ----------------------------------------------------------------- #
# CLEAN DATA
print("Step 2: Cleaning data...")
t0 = time()

for index, article in enumerate(data.content):

  # delete links
  clean_article = re.sub('https?://.+', '', article)
  # delete non-alphabetical and non-space characters
  clean_article = re.sub('[^A-Za-z\s]', '', clean_article)
  # lemmatise
  doc = nlp(clean_article)
  clean_article = " ".join([token.lemma_ for token in doc])
  # update content to clean content
  data.content[index] = clean_article

  print("Cleaned {}/{} article(s). Time elapsed: {:.3f}s.".format(index + 1, N_SAMPLES, time()-t0), end='\r')

print("\n")
print("Done in %0.3fs." % (time() - t0))


# ----------------------------------------------------------------- #
# EXPORT TO CSV FILE
print("Step 3: Writing clean data to csv file...")
t0 = time()

data.to_csv("data-{}-{}.csv".format(YEAR, MONTH), index=False)

print("Done in %0.3fs." % (time() - t0))
