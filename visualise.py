"""
Author: Quang Minh Nguyen and Maida Aizaz
Plot cosine similarity matrices from data files
"""

import json
import itertools
import sys

from numpy.linalg import norm

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

# -------------------------------------------------------------------
# CONSTANTS
MONTHS = ['2016-1', '2016-2', '2016-3', '2016-4', 
    '2016-5', '2016-6',
    '2016-7', '2016-8', '2016-9',
    '2016-10', '2016-11', '2016-12', '2017-1',
    '2017-2', '2017-3', '2017-4', '2017-5', '2017-6']
N_PERIODS = len(MONTHS) - 1


# -------------------------------------------------------------------
# FACILITY FUNCTIONS

def data_from_json(file):
  """
  Import topic/emotion data from json file
  """
  with open(file, "r") as read_file:
    data = json.load(read_file)
  return data

def similarity(data1, data2):
  """
  Generate similarity matrix from topic/emotion data
  """
  matrix = np.zeros((len(data1), len(data2)))
  
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      common_keys = data1[i].keys() & data2[j].keys()
      # divide by 1e5 to avoid overflow
      all1 = np.array(list(data1[i].values()))/1e5
      all2 = np.array(list(data2[j].values()))/1e5
      value1 = np.array([data1[i][key] for key in common_keys])/1e5
      value2 = np.array([data2[j][key] for key in common_keys])/1e5
      matrix[i, j]  = np.inner(value1, value2)/(norm(all1)*norm(all2))
  
  return matrix

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=.3)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')


# -------------------------------------------------------------------
# GET SIMILARITY MATRICES FROM JSON FILES

topic_data = [data_from_json(f'topics/topics-{month}.json') for month in MONTHS]
emotion_data = [data_from_json(f'emotions/emotions-{month}.json') for month in MONTHS]

topic_matrices = []
emotion_matrices = []
for i in range(len(MONTHS)-1):
  # topic
  current = topic_data[i]
  next = topic_data[i+1]
  topic_matrices.append(similarity(current, next))
  # emotion
  current = emotion_data[i]
  next = emotion_data[i+1]
  emotion_matrices.append(similarity(current, next))

# -------------------------------------------------------------------
# GET SYS ARGV
mode = sys.argv[2]


# -------------------------------------------------------------------
# VISUALISE MATRICES

if mode == 'topic_emotion':

  N_PERIODS = 6

  fig, axes = plt.subplots(nrows=2, ncols=N_PERIODS+1, figsize=(20,6)) 

  for index, ax in enumerate(itertools.chain(*axes)): 
    if index == 0: 
      ax.axis('off') 
      ax.text(-0.5, 0.5, "Topic Similarity", size='x-large') 
    elif index == N_PERIODS+1: 
      ax.axis('off') 
      ax.text(-0.5, 0.5, "Emotion Similarity", size='x-large') 
    elif index in range(1, N_PERIODS+1):
      im = ax.matshow(topic_matrices[index-1], cmap='Blues') 
      ax.xaxis.set_ticks_position('none') 
      ax.yaxis.set_ticks_position('none') 
      ax.grid() 
    else:
      im = ax.matshow(emotion_matrices[index-N_PERIODS-2], cmap='Blues') 
      ax.xaxis.set_ticks_position('none') 
      ax.yaxis.set_ticks_position('none') 
      ax.grid()
    if index in range(1, N_PERIODS+1): 
      ax.set_title(f"{MONTHS[index]} and {MONTHS[index+1]}", size='small')

  fig.colorbar(im, ax=axes.ravel().tolist())
  fig.suptitle("Similarity of Topics and Their Emotions over Time") 
  plt.savefig("topic_emotion.jpg")
  plt.cla()

# N_PERIODS = len(MONTHS) - 1


# --------------------------------------------------------------------
# SCATTER PLOTS


if mode == 'scatter':
  DELTA = 0.05

  fig, axes = plt.subplots(nrows=3, ncols=int(N_PERIODS/3)+1, figsize=(30,20)) 

  for index, ax in enumerate(itertools.chain(*axes)): 
    if index >= N_PERIODS:
      ax.axis('off')
      break
    else:
      x = topic_matrices[index].flatten()
      y = emotion_matrices[index].flatten()
      ax.scatter(x, y, alpha=.5)
      ax.set_xlim(-DELTA, 1+DELTA)
      ax.set_ylim(0.2-DELTA, 1+DELTA)    
      ax.set_xlabel('Topic similarity')
      ax.set_ylabel('Emotion similarity')
      ax.set_title(f'{MONTHS[index]} and {MONTHS[index+1]}')

  plt.suptitle('Distribution of Topic and Emotion Similarity')
  plt.savefig('scatter.jpg')
  plt.cla() 


# # --------------------------------------------------------------------
# # COMBINED SCATTER PLOT

if mode == 'combined_scatter':

  x = np.array(topic_matrices).flatten()
  y = np.array(emotion_matrices).flatten()

  DELTA = 0.05
  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.005

  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]

  fig = plt.figure(figsize=(10, 10))

  ax = fig.add_axes(rect_scatter)
  ax_histx = fig.add_axes(rect_histx, sharex=ax)
  ax_histy = fig.add_axes(rect_histy, sharey=ax)

  scatter_hist(x, y, ax, ax_histx, ax_histy)
  ax.set_xlim(-DELTA, 1+DELTA)
  ax.set_ylim(0.2-DELTA, 1+DELTA)    
  ax.set_xlabel('Topic similarity')
  ax.set_ylabel('Emotion similarity')
  plt.suptitle(f'Distribution of Topic and Emotion Similarity ({MONTHS[0]} to {MONTHS[-1]})')
  plt.savefig('combined_scatter.jpg')
  plt.cla()


# --------------------------------------------------------------------
# ANIMATED SIMILARITY DISTRIBUTION

if mode == 'scatter_animation':

  DELTA = 0.05

  x = topic_matrices[0].flatten()
  y = emotion_matrices[0].flatten()
  fig = plt.figure(figsize=(12,12))
  scatter = plt.scatter(x, y, alpha=.5, s=100)
  plt.xlim(-DELTA, 1+DELTA)
  plt.ylim(0.2-DELTA, 1+DELTA)
  plt.xlabel('Topic Similarity')
  plt.ylabel('Emotion Similarity')

  def update(i, topic_matrices, emotion_matrices, scatter):
    x = topic_matrices[i].flatten()
    y = emotion_matrices[i].flatten()
    scatter.set_offsets(np.array([x, y]).T)
    plt.title(f'Topic and Emotion Similarity between Period i={i} and Period i+1={i+1}')
    return scatter,

  ani = animation.FuncAnimation(fig, update, frames=N_PERIODS, 
      fargs=(topic_matrices, emotion_matrices, scatter),
      interval = 200, blit=True)
  ani.save('scatter_animation.gif')
  plt.cla()


# -------------------------------------------------------------------
# ANIMATED TOPIC AND EMOTION SIMILARITY MATRICES

if mode == 'matrix_animation':

  x = emotion_matrices[0]
  y = topic_matrices[0]
  fig, (topic, emotion) = plt.subplots(1, 2, figsize=(20,10))
  plot_topic    = topic.imshow(x, cmap='Blues')
  plot_emotion  = emotion.imshow(y, cmap='Blues')
  plot = [plot_topic, plot_emotion]

  topic.grid()
  emotion.grid()

  def update(i, topic_matrices, emotion_matrices, plot):
    x = topic_matrices[i]
    y = emotion_matrices[i]
    plot[0] = topic.imshow(x, cmap='Blues')
    plot[1] = emotion.imshow(y, cmap='Blues')

    topic.set_title(f'Topic Similarity between Period i={i} and Period i+1={i+1}')
    topic.set_xlabel('Topics of period i+1')
    topic.set_ylabel('Topics of period i')

    emotion.set_title(f'Emotion Similarity between Period i={i} and Period i+1={i+1}')
    emotion.set_xlabel('Topics of period i+1')
    emotion.set_ylabel('Topics of period i')
    return plot

  ani = animation.FuncAnimation(fig, update, frames=N_PERIODS, 
      fargs=(topic_matrices, emotion_matrices, plot),
      interval = 200, blit=True)
  ani.save('matrix_animation.gif')