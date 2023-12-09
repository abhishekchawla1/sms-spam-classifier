#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\spam.csv",encoding='latin1')


# In[3]:


df


# In[4]:


df['v1'].unique()


# In[5]:


df['Unnamed: 2'].unique()


# In[6]:


df.sample(10)


# In[7]:


df.info()


# In[8]:


df['v2'].head(1).values


# Data Cleaning

# In[9]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[10]:


df.rename(columns={'v1':'Target','v2':'text'},inplace=True)


# In[11]:


df


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


l=LabelEncoder()


# In[14]:


df['Target']=l.fit_transform(df['Target'])


# In[15]:


df


# ham=0 and spam=1

# In[16]:


df['Target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df.isnull().sum()


# In[19]:


df.duplicated().any()


# In[20]:


df.duplicated().sum()


# In[21]:


df.drop_duplicates(keep='first',inplace=True)


# In[22]:


df.shape


# Data Analysis

# In[23]:


ax=sns.countplot(x='Target',data=df)
for bars in ax.containers:
    ax.bar_label(bars)


# In[24]:


plt.pie(df['Target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[25]:


import nltk


# In[26]:


df['characters']=df['text'].apply(len)


# In[27]:


df


# In[28]:


df.head(1).values


# In[29]:


df['words']=df['text'].apply(lambda x: len(nltk.word_tokenize(x)))


# In[30]:


df


# In[31]:


df['sentences']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[32]:


df


# In[33]:


df.describe()


# In[34]:


df[df['Target']==0].describe()


# In[35]:


df[df['Target']==1].describe()


# In[36]:


sns.histplot(df[df['Target']==0]['characters'])


# In[37]:


sns.histplot(df[df['Target']==1]['characters'])


# In[38]:


plt.figure(figsize=(12,4))
sns.histplot(df[df['Target']==0]['characters'])
sns.histplot(df[df['Target']==1]['characters'],color='red')


# In[39]:


plt.figure(figsize=(12,4))
sns.histplot(df[df['Target']==0]['words'])
sns.histplot(df[df['Target']==1]['words'],color='red')


# In[40]:


sns.pairplot(data=df,hue='Target')


# In[41]:


sns.heatmap(df.corr(),annot=True)


# Data Preprocessing

# In[42]:


df


# In[43]:


df['text']=df['text'].apply(lambda x: x.lower())


# In[44]:


from nltk.corpus import stopwords


# In[45]:


s=stopwords.words('english')


# In[46]:


s


# In[47]:


import string


# In[48]:


p=string.punctuation


# In[49]:


df['words']


# In[50]:


df['text']=df['text'].apply(lambda x: nltk.word_tokenize(x))


# In[51]:


df


# In[52]:


def transform(x):
    y=[]
    for i in x:
        if i.isalnum():
            y.append(i)
    return y


# In[53]:


df['text']=df['text'].apply(transform)


# In[54]:


df


# In[55]:


def transform2(x):
    list=[]
    for w in x:
        if w not in s:
            list.append(w)
    return list


# In[56]:


df['text']=df['text'].apply(transform2)


# In[57]:


df


# In[58]:


df['text'][20]


# In[59]:


from nltk.stem.porter import PorterStemmer


# In[60]:


ps=PorterStemmer()


# In[61]:


df['text']=df['text'].apply(lambda x: [ps.stem(w) for w in x])


# In[62]:


df


# In[63]:


pip install wordcloud


# In[64]:


from wordcloud import WordCloud


# In[65]:


wc=WordCloud(height=500,width=500,background_color='white',min_font_size=10)


# In[66]:


list=df['text'].iloc[0]


# In[67]:


list


# In[68]:


sentence=' '.join(list)


# In[69]:


sentence


# In[70]:


df['sent']=df['text'].apply(lambda x: ' '.join(x))


# In[71]:


df


# In[72]:


spam_WordCloud=wc.generate(df[df['Target']==1]['sent'].str.cat(sep=' '))


# In[73]:


spam_WordCloud


# In[74]:


plt.figure(figsize=(20,10))
plt.imshow(spam_WordCloud)


# In[75]:


ham_WordCloud=wc.generate(df[df['Target']==0]['sent'].str.cat(sep=' '))


# In[76]:


plt.imshow(ham_WordCloud)


# In[77]:


df


# In[78]:


spam_corpus=[]
for data in df[df['Target']==1]['sent'].tolist():
    for word in data.split():
        spam_corpus.append(word)


# In[79]:


len(spam_corpus)


# In[80]:


from collections import Counter


# In[81]:


s=pd.DataFrame(Counter(spam_corpus).most_common(50))


# In[82]:


s


# In[83]:


ax=sns.barplot(x=s[0],y=s[1],data=s)
plt.xticks(rotation='vertical')
for bars in ax.containers:
    ax.bar_label(bars,rotation='vertical')
plt.show()


# In[84]:


ham_corpus=[]
for data in df[df['Target']==0]['sent'].tolist():
    for word in data.split():
        ham_corpus.append(word)


# In[85]:


h=pd.DataFrame(Counter(ham_corpus).most_common(50))


# In[86]:


h


# In[87]:


sns.barplot(x=h[0],y=h[1],data=h)
plt.xticks(rotation='vertical')
plt.show()


# Model

# In[88]:


df


# In[89]:


from sklearn.feature_extraction.text import CountVectorizer


# In[90]:


cv=CountVectorizer()


# In[91]:


x=cv.fit_transform(df['sent']).toarray()


# In[92]:


x.shape


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


X_train,X_test,y_train,y_test=train_test_split(x,df['Target'].values,test_size=0.2,random_state=42)


# In[95]:


X_train


# In[96]:


y_train


# In[97]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB


# In[98]:


from sklearn.metrics import confusion_matrix,precision_score,accuracy_score


# In[99]:


g=GaussianNB()
b=BernoulliNB()
m=MultinomialNB()


# In[100]:


g.fit(X_train,y_train)
ypred1=g.predict(X_test)
a1=accuracy_score(y_test,ypred1)
p1=precision_score(y_test,ypred1)
c1=confusion_matrix(y_test,ypred1)
print(a1)
print(p1)
print(c1)


# In[101]:


b.fit(X_train,y_train)
ypred2=b.predict(X_test)
a2=accuracy_score(y_test,ypred2)
p2=precision_score(y_test,ypred2)
c2=confusion_matrix(y_test,ypred2)
print(a2)
print(p2)
print(c2)


# In[102]:


m.fit(X_train,y_train)
ypred3=m.predict(X_test)
a3=accuracy_score(y_test,ypred3)
p3=precision_score(y_test,ypred3)
c3=confusion_matrix(y_test,ypred3)
print(a3)
print(p3)
print(c3)


# In[103]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[104]:


t=TfidfVectorizer(max_features=3000)


# In[105]:


X=t.fit_transform(df['sent']).toarray()


# In[106]:


X


# In[107]:


y2=df['Target'].values


# In[108]:


X_train2,X_test2,y_train2,y_test2=train_test_split(X,y2,test_size=0.2,random_state=42)


# In[109]:


g2=GaussianNB()


# In[110]:


g2.fit(X_train2,y_train2)


# In[111]:


y_pred1=g2.predict(X_test2)


# In[112]:


accuracy_score(y_test2,y_pred1)


# In[113]:


precision_score(y_test2,y_pred1)


# In[114]:


confusion_matrix(y_test2,y_pred1)


# In[115]:


m2=MultinomialNB()


# In[116]:


m2.fit(X_train2,y_train2)


# In[117]:


y_pred2=m2.predict(X_test2)


# In[118]:


accuracy_score(y_test2,y_pred2)


# In[119]:


precision_score(y_test2,y_pred2)


# In[120]:


confusion_matrix(y_test2,y_pred2)


# In[121]:


b2=BernoulliNB()


# In[122]:


b2.fit(X_train2,y_train2)


# In[123]:


y_pred3=b2.predict(X_test2)


# In[124]:


confusion_matrix(y_test2,y_pred3)


# In[125]:


precision_score(y_test2,y_pred3)


# In[126]:


accuracy_score(y_test2,y_pred3)


# using multinomial naive bayes and tfidf vectorizer

# In[127]:


import pickle


# In[135]:


pickle.dump(t,open('Vectorizer.pkl','wb'))
pickle.dump(m2,open('MultinomialNaiveBayes','wb'))

