"""
Author: Quang Minh Nguyen and Maida Aizaz
Perform exploratory data analysis and visualise:
- Distribution of publications
- Monthly article counts
- Topics of a chosen month
"""

import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use("fivethirtyeight")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# ----------------------------------------------------------------- #
# CONSTANTS
DURATION    = 1  # topics will be analysed in chunk of 3 months
MONTH       = 12  # month to visualise topics


# ----------------------------------------------------------------- #
# IMPORT DATA FROM FILES
months = [f"2016-{i + 1}" for i in range(12)] + \
    [f"2017-{i + 1}" for i in range(6)]
data = [
    pd.read_csv(f"data/data-{month}.csv")
    [["publication", "content"]] 
    for month in months
]
# n_articles = [len(data[month]) for month in range(len(data))]
# combined_data = pd.concat(
#     data, 
#     axis=0, 
#     ignore_index=True
# )


# ----------------------------------------------------------------- #
# DISTRIBUTION OF PUBLICATIONS
# plt.figure(figsize=(15,15))
# combined_data.publication.value_counts().plot.bar()
# plt.xticks(rotation=25)
# plt.title("Distribution of Publications from 2016.01 to 2017.06")
# plt.savefig("publication-distribution.jpg")
# plt.clf()


# ----------------------------------------------------------------- #
# MONTHLY ARTICLE COUNTS
# plt.bar(months, n_articles)
# plt.xticks(rotation=40)
# plt.title("Number of Articles Each Month from 2016.01 to 2017.06")
# plt.savefig("article-counts.jpg")
# plt.clf()


# ----------------------------------------------------------------- #
# VISUALISE LATENT DIRICHLET ALLOCATION

# bag of words representation
tf_vectorizer = CountVectorizer(
    max_df=.7,  # words occuring in >70% of documents are excluded
    min_df=5,   # words occuring in <5 document are excluded
    stop_words="english")

# latent dirchlet allocation for topic discovery
lda = LatentDirichletAllocation(
    learning_method="online",
    learning_offset=50.0,
    random_state=0)

# create bag of words
tf = tf_vectorizer.fit_transform(data[MONTH].content.values.astype('str'))
# generate latent dirichlet allocation model
lda.fit(tf)
# get words in topics
tf_feature_names = tf_vectorizer.get_feature_names()

fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
axes = axes.flatten()
for topic_index, topic in enumerate(lda.components_):
  top_features_indices = topic.argsort()[: -11 : -1]
  top_features = [tf_feature_names[i] for i in top_features_indices]
  weights = topic[top_features_indices]

  ax = axes[topic_index]
  ax.barh(top_features, weights, height=0.7)
  ax.set_title(f"Topic {topic_index+1}", fontdict={"fontsize": 30})
  ax.invert_yaxis()
  ax.tick_params(axis="both", which="major", labelsize=20)
  for i in "top right left".split():
    ax.spines[i].set_visible(False)
  fig.suptitle("Topics in 2017-01", fontsize=40)

plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
plt.savefig("topics.jpg")