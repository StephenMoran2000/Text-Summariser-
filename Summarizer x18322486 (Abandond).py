#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re


# In[3]:


df = pd.read_csv("tennis_articles.csv")


# In[4]:


df.head()


# In[5]:


############################### MAPPER ###############################
from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list


# In[6]:


#Extract vector representations of words GloVe
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


# In[7]:


#length
len(word_embeddings)


# In[8]:


####################### REDUCER #########################
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


# In[9]:


#Get rid of the stopwords (i.e. is, am, the, of, in, etc.)
nltk.download('stopwords')


# In[10]:



from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[11]:


# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# In[12]:


# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# In[14]:


sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)


# In[15]:


# similarity matrix- define a zero matrix of dimensions (n * n)
sim_mat = np.zeros([len(sentences), len(sentences)])


# In[16]:


#use Cosine Similarity to compute the similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


#initislize Matrix
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


# In[51]:


# create graph the nodes = sentences and the edges = similarity scores between the sentences. 
# i.e the closer to the center , teh more similar it is.
import networkx as nx
import matplotlib.pyplot as plt

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

##print(scores)
pos = nx.spiral_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels = True, node_color="#f86e00")
plt.show()


# In[53]:


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


# In[55]:


# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
#Time taken
import timeit

def test(n):
    return sum(range(n))

n = 10000
loop = 1000

result = timeit.timeit('test(n)', globals=globals(), number=loop)
print(result / loop)
# 0.0002666301020071842


# In[ ]:


# 

