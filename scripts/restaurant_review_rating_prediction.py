#!/usr/bin/env python
# coding: utf-8

# In[31]:


import json
import pandas as pd


#Import Required Module
#!pip install tensorflow
#!pip install keras
#!pip install wordcloud
#!pip install libomp
#!pip install --upgrade libomp
#!pip3 install xgboost
#!pip install --upgrade xgboost #ran this command on terminal on mac OS 'conda install -c conda-forge xgboost'
#!pip install gensim
#!pip install pronouncing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re,string
import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
stops = set(stopwords.words("english"))
punctuation = string.punctuation
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm,decomposition, ensemble
from sklearn.metrics import classification_report,roc_curve,confusion_matrix,auc
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from keras import layers, models, optimizers
from sklearn.decomposition import PCA,TruncatedSVD
import xgboost
import gensim
import warnings
warnings.filterwarnings('ignore')
import pronouncing



# In[32]:


df_business = pd.read_json('yelp_training_set_business.json', lines=True)
print(df_business.head())

df_business.shape


# In[33]:



import json
import pandas as pd

from ast import literal_eval


json_read = pd.read_json('yelp_training_set_review.json',  orient="records",lines=True, chunksize = 10000)


df_review = pd.concat(json_read)
print(df_review.head())
df_review.shape

df = pd.DataFrame(df_review['votes'].values.tolist(), index=df_review.index)
print (df)


# In[34]:


df_user = pd.read_json('yelp_training_set_user.json', lines=True)
print(df_user.head())

df_user.shape


# In[35]:


df_checkin = pd.read_json('yelp_training_set_checkin.json', lines=True)
print(df_checkin.head())



# In[36]:


df_merge1=df_review.merge(df_business,how='left', on='business_id')

df_merge1.head()


# In[37]:


df_merge2=df_merge1.merge(df_checkin,how='left', on='business_id')
df_merge2.head()


# In[38]:


df_merge4=df_merge2.merge(df_user,how='left', on='user_id')
df_merge4.head()


# In[39]:



df_merge3=df_merge4.head(50000)
df_merge3.shape


# In[19]:


#Step6: Plot Word cloud for 1 star rating restaurants
print('\nWord cloud for 1 star rating restaurants\n')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_merge3[df_merge3['stars_x']==1]['name_x']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Step6: Plot Word cloud for 2 star rating restaurants
print('\nWord cloud for 2 star rating restaurants\n')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_merge3[df_merge3['stars_x']==2]['name_x']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Step6: Plot Word cloud for 3 star rating restaurants
print('\nWord cloud for 3 star rating restaurants\n')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_merge3[df_merge3['stars_x']==3]['name_x']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Step6: Plot Word cloud for 4 star rating restaurants
print('\nWord cloud for 4 star rating restaurants\n')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_merge3[df_merge3['stars_x']==4]['name_x']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Step6: Plot Word cloud for 5 star rating restaurants
print('\nWord cloud for 5 star rating restaurants\n')
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df_merge3[df_merge3['stars_x']==5]['name_x']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[20]:



################################## VISUALIZATION #########################################

#Step4: Create a pie chart to show the percentage wise category distribution
print('\nPie Chart:\n')
labels = '1', '2', '3','4','5'
sizes = [17516, 20957, 35363,79878,76193]
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[29]:



import matplotlib.pyplot as plt
num_bins = 5

fig, ax = plt.subplots(figsize = (10,7))

n, bins, patches = ax.hist(df_merge3['stars_x'], num_bins, facecolor='#2b8cbe', alpha=0.8, edgecolor='#000000', linewidth=1)

ax.set_title('Histogram of Ratings', fontsize = 15, pad=15)
ax.set_xlabel('rating')
ax.set_ylabel('frequency')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


x = df_merge3['stars_x']
y = df_merge3['stars_x'].value_counts(ascending=True)
fig, ax = plt.subplots(figsize=(12,10)  )  
width = 0.75 # the width of the bars 
df_merge3['stars_x'].value_counts().plot(kind='bar');
plt.title("Star Rating Distribution")
plt.ylabel('# of businesses', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)
plt.show()
  
#Step4: Create a pie chart to show the percentage wise rating distribution
print('\nPie Chart:\n')
labels = '1', '2', '3','4','5'
sizes = [17516, 20957, 35363,79878,76193]
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[40]:



# bin the data into negative, neutral, and positive values
bins = [0, 2, 4, 6]

bin_names = ['negative', 'neutral', 'positive']

score_bin = pd.Series(df_merge3.stars_x, name = 'score')

score = pd.cut(score_bin, bins, labels=bin_names, right=False)

# number of counts per score

pd.value_counts(score)

df_merge3 = pd.concat([df_merge3, score], axis=1)

df_merge3.head(2)


# In[41]:



# number of counts per score


top_restaurants=df_merge3[(df_merge3['stars_x']==5 ) ]

# top 10 restaurants with most reviews
#top_restaurants_10= top_restaurants.head(10)
top_restaurants=top_restaurants.drop_duplicates(subset=['latitude','longitude','name_x','review_count_x'])
top_restaurants_10 = top_restaurants.sort_values(['review_count_x'], ascending=[0]).head(20)
print(top_restaurants_10.head())
#!pip install folium pandas
import folium
#!pip install --upgrade pandas


top_restaurants_10=top_restaurants_10[['latitude','longitude','name_x','review_count_x']]  

#top_restaurants_10 = top_restaurants_10.sort_values(['review_count'], ascending=[0]).head(20)



center = [33.581867	, -112.241596]
map_USA = folium.Map(location=center, zoom_start=8)
for index, top_restaurants_10 in top_restaurants_10.iterrows():
    location = [top_restaurants_10['latitude'], top_restaurants_10['longitude']]
    folium.Marker(location, popup = f'Latitude:{top_restaurants_10["latitude"]}\n Top rated restaurant name:{top_restaurants_10["name_x"]}').add_to(map_USA)


map_USA


# In[45]:



# number of counts per score

top_rated_restaurants = pd.Series(df_merge3['name_x'])
top_restaurants_counts = pd.value_counts(top_rated_restaurants)

# top 10 restaurants with most reviews
top_restaurants_counts.head(10)


# In[46]:


import matplotlib.pyplot as plt


df_merge3['text_cleaned'] = df_merge3['text'].apply(lambda x: x.split())
df_merge3.head()




from collections import defaultdict
word_freq = defaultdict(int)
for sent in df_merge3['text_cleaned']:
    for i in sent: 
        word_freq[i] += 1
len(word_freq)



sorted(word_freq, key=word_freq.get, reverse=True)[:10]


sentences = df_merge3['text_cleaned']
# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 40   # ignore all words with total frequency lower than this                       
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    


# Initialize and train the model (this will take some time)
from gensim.models import word2vec


model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context)

print("Training finished!")

# save the model for later use. You can load it later using Word2Vec.load()
model_name = "Word_Embedding"
model.save(model_name)



# Get vocabulary count of the model
vocab_tmp = list(model.wv.vocab)
print('Vocab length:',len(vocab_tmp))



from sklearn.metrics.pairwise import cosine_similarity

model.similarity('dish', 'plate')



model.most_similar(positive=['tasty', 'pleased','health','enjoy'], negative=['bad'],topn=20)


from gensim.models import Word2Vec
# Load the trained modelNumeric Representations of Words
model = Word2Vec.load("Word_Embedding")


vocab_tmp = list(model.wv.vocab)
print('Vocab length:',len(vocab_tmp))


# Get distributional representation of each word
X = model[vocab_tmp]


from sklearn import decomposition
# get two principle components of the feature space
pca = decomposition.PCA(n_components=2).fit_transform(X)


good_list = [x for x,y in model.most_similar('great',topn=20)]
bad_list = [x for x,y in model.most_similar('worst',topn=20)]
# good_list = [x for x,y in model.most_similar(positive=['good', 'great','health','sanitary'], negative=['bad'],topn=10)]


# set figure settings
plt.figure(figsize=(15,15))

# save pca values and vocab in dataframe df
df = pd.concat([pd.DataFrame(pca),pd.Series(vocab_tmp)],axis=1)
df.columns = ['x', 'y', 'word']

plt.xlabel("1st principal component", fontsize=14)
plt.ylabel('2nd principal component', fontsize=14)

plt.scatter(x=df['x'], y=df['y'],s=3,alpha=0.3)

good_words = df[df['word'].isin(good_list)]['word']
for i, word in good_words.items():
    plt.annotate(word, (df['x'].iloc[i], df['y'].iloc[i]),fontsize=16,color='green')
   
    
bad_words = df[df['word'].isin(bad_list)]['word']
for i, word in bad_words.items():
    plt.annotate(word, (df['x'].iloc[i], df['y'].iloc[i]),fontsize=13,color='red')


plt.title("PCA Embedding", fontsize=18)


plt.show()


# In[48]:



###################################### FEATURE ENGINEERING ##############################################

# Step 9: NLP/Text based features such as char_count/word_count/punctuation_count
# extract features from text
df_merge3['char_count'] = df_merge3['text'].apply(len)
df_merge3['word_count'] = df_merge3['text'].apply(lambda x: len(x.split()))
df_merge3['word_density'] = df_merge3['char_count'] / (df_merge3['word_count']+1)
df_merge3['punctuation_count'] = df_merge3['text'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
df_merge3['title_word_count'] = df_merge3['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df_merge3['upper_case_word_count'] = df_merge3['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
df_merge3['stopword_count'] = df_merge3['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stops]))


df_merge3['line_count'] = df_merge3['text'].apply(lambda x: len([(line) for line in x.split('\r\n')]))
#print(df['line_count'])
print('\nPrint NLP/Text based features:\n')
print(df_merge3[['char_count', 'word_count', 'word_density', 'punctuation_count', 'title_word_count', 'upper_case_word_count', 'stopword_count','line_count']].head(10))



# In[49]:





counts_df = df_merge3[['score', 'text','word_count']]

# separate by positive and negative reviews
counts_pos = counts_df.loc[(counts_df['score']=='positive')]

counts_neg = counts_df.loc[(counts_df['score']=='negative')]

# create figure
fig, ax = plt.subplots(figsize = (12,10))

sns.boxplot(x=counts_df['score'], y=counts_df['word_count'])

# title
ax.set_title('Number of Words in the Reviews', fontsize = 15, loc = 'left')

# set x axis label
ax.set_xlabel('Sentiment of Review', fontsize = 13)

# set y axis label
ax.set_ylabel('Word Count', fontsize = 13)

# remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.show()


# In[51]:







#Step 12:  Clean text:  no punctuation/all lowercase/remove stop words




#Convert all cases to lower case
df_merge3 = df_merge3.astype(str).apply(lambda x: x.str.lower())
#print('\nFew sample records after converting strings to low case:\n')
#print(df.head())

#Remove the punctuations from the dataframe
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df_merge3['text'] = df_merge3['text'].apply(remove_punctuations)
#print('\nFew sample records after removing punctuations:\n')
#print(df.head())


#Remove stop words from dataframe
df_merge3['text'] = df_merge3['text'].apply(lambda x: ' '.join([item for item in x.split() if item not in stops]))
#print('\nFew sample records after removing stop words:\n')
#print(df.head())

#Apply porter_stemmer on dataframe
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

df_merge3['txt_tokenized']=df_merge3['text'].apply(lambda x : filter(None,x.split(" ")))
df_merge3['txt_stemmed']=df_merge3['txt_tokenized'].apply(lambda x : [porter_stemmer.stem(y) for y in x])
df_merge3['txt_stemmed_sentence']=df_merge3['txt_stemmed'].apply(lambda x : " ".join(x))
print('\nFew sample records after doing cleaning/preprocessing (convert to low case/remove punctuation/remove stopwords/apply porter stemmer:\n')
print(df_merge3.head())



# In[53]:




pos = df_merge3.loc[(df_merge3['score']=='positive')]
neg = df_merge3.loc[(df_merge3['score']=='negative')]

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
stopwords = set(STOPWORDS)
pos_text = " ".join(review for review in pos.txt_stemmed_sentence)

# create figure
fig, ax = plt.subplots(figsize = (12,10))

wordcloud = WordCloud(width=1100, height=800, stopwords=stopwords).generate(pos_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

ax.set_title('Positive Reviews', pad=15, fontsize = 20)
ax.title.set_position([.12, 0])


plt.show()


neg_text = " ".join(review for review in neg.txt_stemmed_sentence)

# create figure
fig, ax = plt.subplots(figsize = (12,10))

wordcloud = WordCloud(width=1100, height=800, stopwords=stopwords).generate(neg_text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

ax.set_title('Negative Reviews', pad=15, fontsize = 20)
ax.title.set_position([.13, 0])


plt.show()


# In[23]:



big_string_proc = ' '.join(df_merge3.txt_stemmed_sentence)
all_words_proc = big_string_proc.split()
print(len(all_words_proc))


# create dictionary of word counts
fdist = FreqDist(all_words_proc)

# convert word counts to dataframe
fdist_df = pd.DataFrame(data=fdist.values(),index=fdist.keys(), columns=['word_count'])
fdist_df = fdist_df.sort_values('word_count',ascending=False)
top_25 = fdist_df.iloc[:25,:]
print(top_25)


# create labels and prettify the plot
plt.figure(figsize=(30,10))
plt.title('Top 25 Words', fontsize=36, pad=15)
plt.ylabel('Word Count', fontsize=30, labelpad=15)
plt.xticks(rotation=55, fontsize=28)
plt.yticks(fontsize=28)
plt.ylim(bottom=100, top=200000)

# plot top 25 words
plt.bar(top_25.index, top_25.word_count, color='purple')

# prepare to save and display
plt.tight_layout()
plt.show()





# In[17]:







from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

# Check most important features in the dataset with chi2 score test

print('\nBelow are the most important features based on Chi2 score:\n')

Tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
tfidf_features = Tfidf.fit_transform(df_merge3['txt_stemmed_sentence'])
tfidf_features.shape

N = 5
Number = 1
for rating in df_merge3['stars_x'].unique():
                features_chi2 = chi2(tfidf_features, df_merge3['stars_x'] == rating)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(Tfidf.get_feature_names())[indices]
                unigrams = [x for x in feature_names if len(x.split(' ')) == 1]
                bigrams = [x for x in feature_names if len(x.split(' ')) == 2]
                print("{}. {} :".format(Number,rating))
                print("\t Unigrams :\n\t. {}".format('\n\t. '.join(unigrams[-N:])))
                print("\t Bigrams :\n\t. {}".format('\n\t. '.join(bigrams[-N:])))
                Number += 1
                
print('\n')              


# In[24]:




#df_merge3.drop('text',axis='columns', inplace=True)
print(list(df_merge3.columns))

df_merge3.drop('char_count',axis='columns', inplace=True)
df_merge3.drop('word_count',axis='columns', inplace=True)
df_merge3.drop('word_density',axis='columns', inplace=True)
df_merge3.drop('punctuation_count',axis='columns', inplace=True)
df_merge3.drop('title_word_count',axis='columns', inplace=True)
df_merge3.drop('upper_case_word_count',axis='columns', inplace=True)
df_merge3.drop('stopword_count',axis='columns', inplace=True)
df_merge3.drop('line_count',axis='columns', inplace=True)
df_merge3.drop('txt_tokenized',axis='columns', inplace=True)
df_merge3.drop('txt_stemmed',axis='columns', inplace=True)


# In[25]:


########################  MODEL BUILDING AND EVALUATIONS ###########################################          

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_merge3['txt_stemmed_sentence'], df_merge3['stars_x'])


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

################### Create features from text #########################

# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df_merge3['txt_stemmed_sentence'])

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df_merge3['txt_stemmed_sentence'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df_merge3['txt_stemmed_sentence'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df_merge3['txt_stemmed_sentence'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)






# In[23]:


######## Test with Truncated SVD technique to check the optimum number of components/features
# Dimensionality reduction


for num_components in range(100):
                    svd = TruncatedSVD(n_components=num_components, random_state=42)
                    X_svd = svd.fit_transform(xtrain_tfidf)
                    print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}", 'number of components:',num_components)

# Check explained variance ratio for 300 components/features
svd = TruncatedSVD(n_components=300, random_state=42)
X_svd = svd.fit_transform(xtrain_tfidf)
#print("\nTotal variance explained with 300 components: {np.sum(svd.explained_variance_ratio_):.2f}")
print(f"Total variance explained: {np.sum(svd.explained_variance_ratio_):.2f}", 'number of components:','300')


# In[26]:


############################### Train, Build and Evaluate the model ####################################3
#Write a function to train the model classifier that will calculate the accuracy 
#The function will also plot the confusion matrix and will print the classification report

def train_model(model_name,classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    cf_matrix=confusion_matrix(valid_y, predictions)
    print('\nConfusion matrix for' ,model_name,'is:\n')
    print(confusion_matrix(valid_y, predictions))
    print('\nClassification report for',model_name,'is:\n')
    print(classification_report(valid_y, predictions,target_names=['1','2','3','4','5']))
    index = ['0','1','2','3','4']  
    columns = ['1','2','3','4','5']  
    cm_df = pd.DataFrame(cf_matrix,columns,index)                      
    plt.figure(figsize=(5.5,4))  
    sns.heatmap(cm_df, annot=True,cmap="viridis" ,fmt='g')
    plt.xticks([0,1,2,3,4])
    plt.yticks([0,1,2,3,4])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(model_name)
    plt.show()
    
    return metrics.accuracy_score(predictions, valid_y)



# In[19]:


# Naive Bayes on Count Vectors
accuracy_nb_cv = train_model("Naive Bayes  Count Vectors:",naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print( "Accuracy of Naive Bayes, Count Vectors Model is: ", "{:.2%}".format(accuracy_nb_cv))

# Naive Bayes on Word Level TF IDF Vectors
accuracy_nb_tfidf = train_model("NB, WordLevel TF-IDF: ",naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print( "Accuracy of Naive Bayes, WordLevel TF-IDF Model is: ", "{:.2%}".format(accuracy_nb_tfidf))

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy_nb_ngtfidf = train_model("NB, N-Gram Vectors: ",naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print( "Accuracy of Naive Bayes, N-Gram Vectors Model is: ", "{:.2%}".format(accuracy_nb_ngtfidf))


# Naive Bayes on Character Level TF IDF Vectors
accuracy_nb_ctfidf = train_model("NB, CharLevel Vectors: ",naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print( "Accuracy of Naive Bayes, CharLevel Vectors Model is: ", "{:.2%}".format(accuracy_nb_ctfidf))


# In[20]:







# Linear Classifier on Count Vectors
accuracy_lr_cv = train_model("LR, Count Vectors: ",linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print( "Accuracy of Liner Regression Count Vectors Model is: ", "{:.2%}".format(accuracy_lr_cv))

# Linear Classifier on Word Level TF IDF Vectors
accuracy_lr_tfidf = train_model("LR, WordLevel TF-IDF: ",linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print( "Accuracy of Liner Regression TFIDF Model is: ", "{:.2%}".format(accuracy_lr_tfidf))

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy_lr_ngtfidf = train_model("LR, N-Gram Vectors: ",linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print( "Accuracy of Liner Regression N-Gram Level Model is: ", "{:.2%}".format(accuracy_lr_ngtfidf))

# Linear Classifier on Character Level TF IDF Vectors
accuracy_lr_ctfidf = train_model("LR, CharLevel TF-IDF: ",linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print( "Accuracy of Liner Regression Char Level TFIDF Model is: ", "{:.2%}".format(accuracy_lr_ctfidf))



# In[18]:


# SVM on Ngram Level TF IDF Vectors
accuracy_svm_ngtfidf = train_model("SVM, N-Gram Vectors: ",svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print( "Accuracy of SVM, N-Gram Vectors Model is: ", "{:.2%}".format(accuracy_svm_ngtfidf))




# In[19]:


# RF on Count Vectors
accuracy_rf_cv = train_model("RF, Count Vectors: ",ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print( "Accuracy of Random Forest Count Vector Model is: ", "{:.2%}".format(accuracy_rf_cv))


# In[20]:


# RF on Word Level TF IDF Vectors
accuracy_rf_tfidf = train_model("RF, WordLevel TF-IDF: ",ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print( "Accuracy of Random Forest word level Model is: ", "{:.2%}".format(accuracy_rf_tfidf))



# In[21]:


# Extereme Gradient Boosting on Count Vectors
accuracy_xgb_cv = train_model("Xgb, Count Vectors: ",xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print( "Accuracy of Xgradient boost count vector Model is: ", "{:.2%}".format(accuracy_xgb_cv))


# In[22]:


# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy_xgb_tfidf = train_model("Xgb, WordLevel TF-IDF: ",xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print( "Accuracy of Xgradient boost TFIDF vector Model is: ", "{:.2%}".format(accuracy_xgb_tfidf))




# In[23]:


# Extereme Gradient Boosting on Character Level TF IDF Vectors
accuracy_xgb_cltfidf = train_model("Xgb, CharLevel Vectors: ",xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print( "Accuracy of Xgradient boost char level vector Model is:", "{:.2%}".format(accuracy_xgb_cltfidf))





# In[37]:




train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_merge3['txt_stemmed_sentence'], df_merge3['stars_x'])


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)



# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df_merge3['txt_stemmed_sentence'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x).toarray()
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x).toarray()


from keras.models import Sequential
from keras import layers

input_dim = xtrain_tfidf_ngram.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
model.summary()



history = model.fit(xtrain_tfidf_ngram, train_y, epochs=100, verbose=False, validation_data=(xvalid_tfidf_ngram, valid_y)  ,batch_size=10)

loss, accuracy = model.evaluate(xtrain_tfidf_ngram, train_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(xvalid_tfidf_ngram, valid_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[35]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_merge3['txt_stemmed_sentence'], df_merge3['stars_x'])


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(df_merge3['txt_stemmed_sentence'])

X_train = vectorizer.transform(train_x)
X_test  = vectorizer.transform(valid_x)




# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print(X_train.shape)
print(X_test.shape)
print(train_y.shape)
print(valid_y.shape)




def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(5, activation="softmax")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(X_train.shape[1])
accuracy_neural_network = train_model("Neural Network Ngram Level TF IDF Vector Model",classifier, X_train, train_y, X_test, is_neural_net=True)
print( "Accuracy of Neural Network Ngram Level TF IDF Vector Model is:", "{:.2%}".format(accuracy_neural_network))


# In[39]:


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_merge3['txt_stemmed_sentence'], df_merge3['stars_x'])


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)



# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df_merge3['txt_stemmed_sentence'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x).toarray()
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x).toarray()


from keras.models import Sequential
from keras import layers



print(X_train.shape)
print(X_test.shape)
print(train_y.shape)
print(valid_y.shape)



def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(10, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(5, activation="softmax")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy_neural_network_ngram = train_model("Neural Network Ngram Level TF IDF Vector Model",classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print( "Accuracy of Neural Network Ngram Level TF IDF Vector Model is:", "{:.2%}".format(accuracy_neural_network_ngram))


# In[44]:


#################### Word2vec word embedding using xgboost classifier #############################3

test_size = 0.3
random_state = 1234

X_train, X_test, y_train, y_test = model_selection.train_test_split(df_merge3['txt_stemmed_sentence'], df_merge3['stars_x'], test_size=test_size, random_state=random_state)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec


class GensimWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.
    gensim's own gensim.sklearn_api.W2VTransformer doesn't support out of vocabulary words,
    hence we roll out our own.
    All the parameters are gensim.models.Word2Vec's parameters.
    https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    """

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
                 ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                 callbacks=(), max_final_vocab=None):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.max_final_vocab = max_final_vocab

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            sentences=X, corpus_file=None,
            size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count,
            max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed,
            workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
            negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words,
            compute_loss=self.compute_loss, callbacks=self.callbacks,
            max_final_vocab=self.max_final_vocab)
        return self

    def transform(self, X):
        X_embeddings = np.array([self._get_embedding(words) for words in X])
        return X_embeddings

    def _get_embedding(self, words):
        valid_words = [word for word in words if word in self.model_.wv.vocab]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                embedding[idx] = self.model_.wv[word]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.size)




gensim_word2vec_tr = GensimWord2VecVectorizer(size=50, min_count=3, sg=1, alpha=0.025, iter=10)
xgb = xgboost.XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
w2v_xgb = Pipeline([('w2v', gensim_word2vec_tr),  ('xgb', xgb)])
#w2v_xgb

w2v_xgb.fit(X_train, y_train)




y_train_pred = w2v_xgb.predict(X_train)
print('\nConfusion Matrix for word2vec xgboost model:\n')
y_test_pred = w2v_xgb.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
accuracy_w2v_xgb=metrics.accuracy_score(y_test, y_test_pred)
print( "Accuracy of XGBoost word2vec model classifier is:", "{:.2%}".format(metrics.accuracy_score(y_test, y_test_pred)))
cf_matrix=confusion_matrix(y_test, y_test_pred)
index = ['0','1','2','3','4']  
columns = ['0','1','2','3','4']  
cm_df = pd.DataFrame(cf_matrix,columns,index)                      
plt.figure(figsize=(5.5,4))  
sns.heatmap(cm_df, annot=True,cmap="viridis" ,fmt='g')
plt.xticks([0,1,2,3,4])
plt.yticks([0,1,2,3,4])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("word2vec xgboost")
plt.show()

print('\n Classification report for word2vec xgboost model:\n')
print(classification_report(y_test, y_test_pred,target_names=['1','2','3','4','5' ]))


# In[27]:


# Function to Plot ROC curve for multi-class classification using different models


def plot_roc(model_name,clf,xtrain_count,y_train,xvalid_count,y_test):
            #clf = MultinomialNB(alpha=.01)
    clf.fit(xtrain_count, y_train)
    pred = clf.predict(xvalid_count)
    pred_prob = clf.predict_proba(xvalid_count)
    print('\n',model_name,'\n')
            # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}

    n_class = 5
    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
        print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
              # plotting    
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--',color='blue', label='Class 4 vs Rest')
    plt.plot(fpr[4], tpr[4], linestyle='--',color='blue', label='Class 5 vs Rest')
    plt.title(model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best') 
    plt.show()
    
    

#################  Plot ROC Curve for  4 different models ######################################
plot_roc("Multiclass ROC Curve for Naive Bayes Classifier",MultinomialNB(alpha=.01),xtrain_count,train_y,xvalid_count,valid_y)
plot_roc("Multiclass ROC Curve for Linear Regression classifier",OneVsRestClassifier(LogisticRegression()),xtrain_count,train_y,xvalid_count,valid_y)
#plot_roc("Multiclass ROC curve for SVM classifier",OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,random_state=0)),xtrain_tfidf_ngram,train_y,xvalid_tfidf_ngram,valid_y)
plot_roc("Multiclass ROC curve for Random Foreset Classifier ",RandomForestClassifier(random_state=123),xtrain_count,train_y,xvalid_count,valid_y)




