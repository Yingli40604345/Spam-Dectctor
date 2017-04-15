
# coding: utf-8

# In[8]:

import nltk


# In[3]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection',encoding="utf8")]


# In[9]:

print(len(messages))


# In[10]:

messages[0]


# In[11]:

messages[50]


# In[12]:

for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[13]:

import pandas as pd


# In[14]:

messages=pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])


# In[15]:

messages


# In[16]:

messages.describe()


# In[17]:

messages.groupby('label').describe()


# In[19]:

messages['length']=messages['message'].apply(len)


# In[20]:

messages.head()


# In[21]:

import matplotlib.pyplot as plt
plt.rcParams['patch.force_edgecolor']=True
import seaborn as sns


# In[22]:

get_ipython().magic('matplotlib inline')


# In[23]:

messages['length'].plot.hist(bins=150)


# In[24]:

messages['length'].describe()


# In[25]:

messages[messages['length']==910]['message'].iloc[0]


# In[26]:

messages.hist(column='length',by='label',bins=60,figsize=(12,4))


# In[27]:

import string


# In[32]:

from nltk.corpus import stopwords


# In[52]:

def text_process(mess):
    '''
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    '''
    nopunc=[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[53]:

messages.head()


# In[54]:

messages['message'].head(5).apply(text_process)


# In[56]:

from sklearn.feature_extraction.text import CountVectorizer


# In[57]:

bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[58]:

print(len(bow_transformer.vocabulary_))


# In[59]:

mess4=messages['message'][3]


# In[60]:

print(mess4)


# In[61]:

bow4=bow_transformer.transform([mess4])


# In[62]:

print(bow4)


# In[63]:

print(bow4.shape)


# In[64]:

bow_transformer.get_feature_names()[4068]


# In[65]:

bow_transformer.get_feature_names()[9554]


# In[70]:

messages_bow=bow_transformer.transform(messages['message'])


# In[71]:

print('Shape of Sparse Matrix: ',messages_bow.shape)


# In[72]:

#non zeros
messages_bow.nnz


# In[76]:

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))


# In[77]:

from sklearn.feature_extraction.text import TfidfTransformer


# In[78]:

tfidf_transformer=TfidfTransformer().fit(messages_bow)


# In[79]:

tfidf4=tfidf_transformer.transform(bow4)


# In[80]:

print(tfidf4)


# In[81]:

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[82]:

messages_tfidf=tfidf_transformer.transform(messages_bow)


# In[84]:

#naive base classfier
from sklearn.naive_bayes import MultinomialNB


# In[85]:

spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['label'])


# In[89]:

spam_detect_model.predict(tfidf4)


# In[88]:

spam_detect_model.predict(tfidf4)[0]


# In[90]:

messages['label'][3]


# In[92]:

all_pred=spam_detect_model.predict(messages_tfidf)


# In[93]:

all_pred


# In[94]:

#split test and training


# In[95]:

from sklearn.cross_validation import train_test_split


# In[96]:

msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['label'],test_size=0.3)


# In[97]:

msg_train


# In[98]:

#pipeline


# In[99]:

from sklearn.pipeline import Pipeline


# In[100]:

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# In[101]:

pipeline.fit(msg_train,label_train)


# In[102]:

predictions=pipeline.predict(msg_test)


# In[103]:

from sklearn.metrics import classification_report


# In[104]:

print(classification_report(label_test,predictions))


# In[106]:

#96% accuracy


# In[108]:

from sklearn.ensemble import RandomForestClassifier


# In[109]:

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])


# In[110]:

pipeline.fit(msg_train,label_train)
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))


# In[ ]:



