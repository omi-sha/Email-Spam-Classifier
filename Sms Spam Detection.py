#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('spam.csv',encoding='latin1')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df.shape


# In[9]:


df.sample(5)


# In[10]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[12]:


df['target']=encoder.fit_transform(df['target'])


# In[13]:


df.duplicated().sum()


# In[14]:


df=df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# #EDA

# In[16]:


df['target'].value_counts()


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[19]:


import nltk


# In[20]:


get_ipython().system('pip install nltk')


# In[21]:


nltk.download('punkt')


# In[22]:


df['num_characters']=df['text'].apply(len)


# In[23]:


df.head()


# In[24]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[25]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df[['num_characters','num_words','num_sentences']].describe()


# In[28]:


df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[29]:


df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[30]:


import seaborn as sns


# In[31]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[32]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[33]:


sns.heatmap(df.corr(),annot=True)


# 3. Data Preprocessing
# Lower case
# Tokenization
# Removing special characters
# Removing stop words and punctuation
# Stemming

# In[34]:


nltk.download('stopwords')


# In[35]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[36]:


import string
string.punctuation


# In[37]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[38]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[39]:


df['transformed_text']=df['text'].apply(transform_text)


# In[40]:


df.head()


# In[41]:


get_ipython().system('pip install wordcloud')


# In[42]:


import matplotlib.font_manager as font_manager

# Get a list of available TrueType fonts
ttf_fonts = font_manager.findSystemFonts()

# Print the list of TrueType fonts
for font in ttf_fonts:
    print(font)


# In[43]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[44]:


len(spam_corpus)


# In[45]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[46]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[47]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
cv = CountVectorizer()

# Convert the 'transformed_text' column to a list
text_list = df['transformed_text'].tolist()

# Fit and transform the text data
X = cv.fit_transform(text_list)
X = X.toarray()


# In[49]:


y = df['target'].values


# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[52]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[53]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[54]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[55]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[56]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[58]:


get_ipython().system(' pip install xgboost')


# In[76]:


from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Define the classifiers
svc = SVC()
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier()
lrc = LogisticRegression()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
bc = BaggingClassifier()
etc = ExtraTreesClassifier()
gbdt = GradientBoostingClassifier()
xgb = XGBClassifier()

# Create a dictionary of classifiers
clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

# Define the train_classifier function
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    return accuracy, precision

# Perform training and evaluation
accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    
    print("For", name)
    print("Accuracy:", current_accuracy)
    print("Precision:", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[ ]:




