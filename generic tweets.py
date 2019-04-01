
# coding: utf-8

# In[1]:


#All the required primary libraries are imported and additional required libraries would be imported throughtout the code flow.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys
import random
import html
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')


# In[2]:


#Reading and visualizing the generic tweets which has the sentiment polarity
generic_tweets = open('generic_tweets.txt', 'r')        
data = generic_tweets.read()   #Read Every Charater till the End of the Line
generic_data = [i for i in data.splitlines() if i.strip()!=''] #Strip any leading and trailing white spaces
generic_data


# In[3]:


#Reading and visualizing the US_Airline_tweets 
airlines_tweets = pd.read_csv("US_airline_tweets.csv")        
airlines_tweets['text'][1]


# ## 1. Data Cleaning
# 

# In[4]:


#data_cleaning function will perform all the requirements as stated in the description above
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def data_cleaning(tweets):
    #Using Regular expression to Clean out tweets
    
    #All html tags and attributes (i.e., /<[^>]+>/) are removed.
    tags = re.compile(r'<.*?>')
    tweets = tags.sub('', tweets)
    
    # Html character codes (i.e., &...;) are replaced with an ASCII equivalent.
    #Convert all named and numeric character references (e.g. &gt;, &#62;, &x3e;) in the strings 
    #to the corresponding unicode characters
    
    tweets = html.unescape(tweets)
    
    # Remove URLs
    url = re.sub(r"http\S+", "", tweets)
    tweets = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(twitter.[^\s])|(instagram.[^\s])|(bit.[^\s])|(pic.[^\s])|(http?://[^\s]+))|','', url)
    
    
    # Remove extra white spaces
    tweets = re.sub('[\s]+', " ", tweets)
    
    # Keep only letters
    tweets = re.sub("[^a-zA-Z]", " ", tweets)
    
    # Convert to lowercase.
    tweets = tweets.lower()
    
    # Remove stopwords
    #Reading Stop_words text file given in the assingment 
    #split gives a list, in which each word is a string
    stopwords = open('stop_words.txt', 'r').read().split()
        
    tweets = tweets.split()  
    
    # " ".join return a string which is the concatination of the strings which are not in the stop words 
    # and the sepeartor between the elements is white space (" ")
    tweets = " ".join(w for w in tweets if not w in stopwords)  
  
    #performing lemmatization
    tweets = " ".join([wnl.lemmatize(i) for i in tweets.split()])

    tweets = "".join(tweets)
    #data_cleaning function takes each tweet and returns cleaned tweets
    return tweets

#Test to check if the data cleaning produces result as expected
random_test1 = "'@VirginAmerica it\'s really can't tell aggressive to blast obnoxious entertainment in your guests\' faces &amp; they have little recourse'"
random_test_out = data_cleaning(random_test1)
#Output shows data_cleaning has produced expected returns for the given sample string
random_test_out


# In[5]:


#Apply data_cleaning function to the US airlines tweets data
#Visualizing cleaned individual tweets
for index, row in airlines_tweets.iterrows():
    print (data_cleaning(airlines_tweets['text'][index]))


# ## 2. Exploratory Analysis

# In[6]:


#This cell contains the function to determine the Airline based on each tweet from the airlines tweet data
def find_airlines(text):
    tweet = data_cleaning(text)
    if 'virginamerica' in tweet:
        return('Virgin America Airlines')
    elif 'united' in tweet:
        return('United Airlines')
    elif 'southwestair' in tweet:
        return('Southwest Airlines')
    elif 'jetblue' in tweet:
        return('Jetblue Airlines')
    elif 'usairways' in tweet:
        return('US Airways')
    elif 'americanair' in tweet:
        return('American Airlines')
    else:
        return('Unknown Classification')
    
#Test find_airlines function
test_func = "'@VirginAmerica it\'s really can't tell aggressive to blast obnoxious entertainment in your guests\' faces &amp; they have little recourse'"
print(find_airlines(test_func))


# In[7]:


#In this cell, am going to iterate over each tweets to find the airlines


#Applying find airlines function on the airlines data
airlines_tweets['Airlines'] = airlines_tweets['text'].apply(find_airlines)
airlines_tweets


# In[8]:


#Visualizing the classification of tweets for the given US airline dataset.  Visualizing only the classified tweets
#and the tweets which wasn't classified, are removed.
airline_classified_df = airlines_tweets[airlines_tweets.Airlines.str.contains("Unknown Classification") == False]
airline_classified_df['Airlines'].value_counts().plot(kind='bar')   
plt.xlabel('Airlines')
plt.ylabel('Number of Tweets')
plt.title('Twitter Analysis on US Airlines Data')


# In[9]:


#Visualizing the Split of sentiment on tweets for the given US airline dataset
airline_classified_df['sentiment'].value_counts().plot(kind='bar')   
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Split of sentiment on US Airlines Data')


# In[10]:


from collections import Counter
#In this cell, we will print and visualize the split of negative and positive tweets for each airlines

count_v = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='Virgin America Airlines'])
count_u = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='United Airlines'])
count_s = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='Southwest Airlines'])
count_j = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='Jetblue Airlines'])
count_us = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='US Airways'])
count_a = Counter(airline_classified_df.sentiment[airline_classified_df.Airlines=='American Airlines'])

#Printing the positive and negative tweets of each airlines
print("Positive sentiment on Virgin airlines",count_v['positive']/(count_v['positive']+count_v['negative'])*100)
print("Positive sentiment on United Airlines",count_u['positive']/(count_u['positive']+count_u['negative'])*100)
print("Positive sentiment on Southwest Airlines",count_s['positive']/(count_s['positive']+count_s['negative'])*100)
print("Positive sentiment on Jetblue Airlines",count_j['positive']/(count_j['positive']+count_j['negative'])*100)
print("Positive sentiment on US Airways",count_us['positive']/(count_us['positive']+count_us['negative'])*100)
print("Positive sentiment on American Airlines",count_a['positive']/(count_a['positive']+count_a['negative'])*100)

positives = (count_v['positive'],count_u['positive'],count_s['positive'],count_j['positive'],count_us['positive'],count_a['positive'])
negatives = (count_v['negative'],count_u['negative'],count_s['negative'],count_j['negative'],count_us['negative'],count_a['negative'])

N = 6
width = 0.5
ind = np.arange(N)
fig, ax = plt.subplots()
rects1  = plt.bar(ind,positives, width, color='r')
rects2  = plt.bar(ind,negatives, width, color='y', bottom = positives)

# add some text for labels, title and axes ticks
ax.set_ylabel('Number of tweets')
plt.xlabel('Airlines')
ax.set_title('Comparing Postive tweets against Negative tweets')
ax.set_xticks(ind)
ax.set_xticklabels(('Virgin\nAmerica\nAirlines','United\nAirlines','Southwest\nAirlines',
                'Jetblue\nAirlines','US\nAirways','American\nAirlines'))

ax.legend((rects1[0], rects2[0]), ('Positives', 'Negatives'))

#Labelling the bar chart
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                int(height),
                ha='center', va='baseline')

autolabel(rects1)
autolabel(rects2)

#Visualizing the positive and negative tweets of each airlines
plt.show()


# In[11]:


#Split of sentiment on the US airline twitter data for 2015
airline_classified_df['sentiment'].value_counts()


# In[12]:


#Total tweet counts for individual airlines are printed out
airline_classified_df['Airlines'].value_counts()


# In[13]:


#Following few cells would try to visualize the generic data
#First step would be to load the generic data into a data frame
generic_df = pd.read_csv('generic_tweets.txt', header=0, names=["class","id","date","query","user","text"])
generic_df.head()


# In[14]:


#A function is defined to label the emotions as Positive and Negative based on the class column
def add_label(class_value):
    if class_value==4:
        return ('positive')
    elif class_value==0:
        return('negative')
    else:
        return('Undetermined')

#Applying add_label function on the generic data
generic_df['label']= generic_df['class'].apply(add_label)
generic_df.head()


# In[15]:


#Total counts for the given two emotion types are printed out
generic_df['label'].value_counts()


# In[16]:


#Visualizing the emotions in terms of a bar plot, this denotes that there is a 50:50 split on the dataset for the given
#2 emotion types
generic_df['label'].value_counts().plot(kind='bar')   
plt.xlabel('Emotion')
plt.ylabel('Number of Tweets')
plt.title('Twitter Analysis on generic data')


# ## 3. Model Preparation
# 

# In[17]:


#Cleaning the generic tweets dataset using the data cleaning function
normalize_text = np.vectorize(data_cleaning)
processed_generic_data = normalize_text(generic_df.text.tolist())
processed_generic_data


# In[18]:


#We are going to use TF_IDF feature model to convert the features to numeric vectors.  TF-IDF model prevents
#overshadowing of more frequent features in a large dataset
from sklearn.feature_extraction.text import TfidfVectorizer

#The vectorizer just uses the top 2000 most frequently occuring features in the given generic dataset
vectorizer = TfidfVectorizer(min_df=0., max_df=1., use_idf=True, max_features=2000)
vec_matrix = vectorizer.fit_transform(processed_generic_data)
vec_matrix = vec_matrix.toarray()

#Printing the numeric feature vectors for each individual tweets, and since we considered only the top 2000 features,
#we are left with a feature matrix of size 200000 rows × 2000 columns
vocab = vectorizer.get_feature_names()
pd.DataFrame(np.round(vec_matrix, 2), columns=vocab)

#That concludes the requirement of section 3


# In[19]:


#Taking X as the feature vectors and Y as the sentiment type
X = vec_matrix[0:len(generic_df)]    # features for overall data
y = generic_df.label                # targets (sentiment values) for overall data


# In[20]:


#import train_test_split
from sklearn.cross_validation import train_test_split

#split the features and targets of classified data into training (70%) and test data (30%) sets
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=0)  


# In[21]:


# initiate a logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() 


# ## 4. Model Implementation
# 

# In[22]:


#Train the model using the training set from the generic data
classifier.fit(X_train, y_train)


# In[23]:


#Predict the sentiment for the test data and check the accuracy of the regression model
from sklearn.metrics import accuracy_score
y_pred_train = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_train))

#Importing confusion matrix and visualizing the spread of true negatives and positives
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_train)
cm


# In[24]:


#Next, we clean each tweets in the US airline tweets data set
processed_airline_data = normalize_text(airline_classified_df.text.tolist())
actual_sentiment = airline_classified_df.sentiment


#Use the model created using generic tweets dataset on the US airline dataset
airline_pred = vectorizer.transform(processed_airline_data) 
pred_airline = classifier.predict(airline_pred)
print("Model prediction accuracy on Airline tweets using logistic regression",round(accuracy_score(actual_sentiment, pred_airline)*100),"%")


# In[25]:


#Here, we filter out the positive sentiments from the US airlines data
neg_airline_df = airline_classified_df[airline_classified_df.sentiment.str.contains("negative") == True]
neg_airline_df.head()


# In[26]:


#Cleaning the negative tweets
processed_neg_airline_data = normalize_text(neg_airline_df.text.tolist())
processed_neg_airline_data


# In[27]:


#We are going to use TF_IDF feature model to convert the features to numeric vectors.
vectorizer1 = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
vec_matrix1 = vectorizer1.fit_transform(processed_neg_airline_data)
vec_matrix1 = vec_matrix1.toarray()

#Printing the numeric feature vectors for each individual tweets, feature matrix size = 200000 rows × 2000 columns
vocab1 = vectorizer1.get_feature_names()
pd.DataFrame(np.round(vec_matrix1, 2), columns=vocab1)


# In[28]:


#Taking X as the feature vectors and Y as the negative reason
X1 = vec_matrix1[0:len(neg_airline_df)] 
y1 = neg_airline_df.negative_reason  

#Spliting the negative tweets in 70%training and 30%testing sets
X1_train, X1_test, y1_train, y1_test  = train_test_split(X1, y1, test_size=0.3, random_state=0) 


# In[29]:


#We now create a multi-class logistic regression model and use Newton's method to find better approximations on the
#class (negative reasons).  
#For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss.
multi_classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg') 

#Training the multi-class model
multi_classifier.fit(X1_train, y1_train)

#Predicting the test data using the multi-class model
y1_pred_train = multi_classifier.predict(X1_test)

#Printing the accuracy score of the model
print(accuracy_score(y1_test, y1_pred_train))


# In[30]:


#We now use classification report to print out the precision of the prediction
#The reported averages include micro average (averaging the total true positives, 
#false negatives and false positives), macro average (averaging the unweighted mean per label),
#weighted average (averaging the support-weighted mean per label) and 
#sample average (only for multilabel classification)
from sklearn.metrics import classification_report
print(classification_report(y1_test, y1_pred_train))


# In[31]:


#Here we use the confusion matrix to show the actual and predicted counts.  The diagonal of the confusion matrix shows
#the true matchings
cm_df = pd.DataFrame(confusion_matrix(y1_test, y1_pred_train, 
                       labels=['Bad Flight', 'Cant Tell', 'Late Flight', 
                               'Customer Service Issue', 'Flight Booking Problems',
                               'Lost Luggage','Flight Attendant Complaints','Cancelled Flight',
                               'Damaged Luggage','longlines']),
                   index=['true:Bad Flight','true:Cant Tell','true:Late Flight',
                          'true:Customer Service Issue','true:Flight Booking Problems',
                          'true:Lost Luggage','true:Flight Attendant Complaints','true:Cancelled Flight',
                          'true:Damaged Luggage','true:longlines'], columns=['pred:Bad Flight','pred:Cant Tell','pred:Late Flight',
                          'pred:Customer Service Issue','pred:Flight Booking Problems',
                          'pred:Lost Luggage','pred:Flight Attendant Complaints','pred:Cancelled Flight',
                          'pred:Damaged Luggage','pred:longlines'])

cm_df


# ### 1. Classification using Naive Bayes

# In[32]:


#For first bonus part, we are going to use Naive bayes, Naive bayes classifier (nb) will be used to 
#train and test the model
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)


# In[33]:


#validate the nb model using test data and show accuracy score
y_pred_train_nb = classifier_nb.predict(X_test)
print(accuracy_score(y_test, y_pred_train_nb))


# In[34]:


#validate the nb model trained on generic data on the airline data and print the accuracy
pred_airline_nb = classifier_nb.predict(airline_pred.toarray())
print("GaussianNB Model prediction accuracy on Airline tweets",round(accuracy_score(actual_sentiment, pred_airline_nb)*100),"%")


# ### 2. Classification using decision tree

# In[35]:


#For second bonus part, we are going to use decision tree classification, decision tree classifier (dtc) will be used to 
#train and test the model
from sklearn.tree import DecisionTreeClassifier

#criterion is a function to measure the quality of split and 'entropy' for information gain
classifier_dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dtc.fit(X_train, y_train)


# In[36]:


#validate the dtc model using test data and show accuracy score
y_pred_train_dtc = classifier_dtc.predict(X_test)
print(accuracy_score(y_test, y_pred_train_dtc))


# In[37]:


#validate the dtc model trained on generic data on the airline data and print the accuracy
pred_airline_dtc = classifier_dtc.predict(airline_pred)
print("DTC Model prediction accuracy on Airline tweets",round(accuracy_score(actual_sentiment, pred_airline_dtc)*100),"%")

