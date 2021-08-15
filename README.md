
# Sentiment Analysis on Amazon Customer Feedback using Python

Sentiment analysis, also refers as opinion mining, is a sub machine learning task where we want to determine the general sentiment of a given document. Using machine learning techniques and natural language processing we can extract the subjective information of a document and try to classify it according to its polarity such as positive, neutral or negative. It is a really useful analysis since we could possibly determine the overall opinion about selling objects, or predict stock markets for a given company like, if most people think positive about it, possibly its stock markets will increase, and so on. Sentiment analysis is actually far from to be solved since the language is very complex (objectivity/subjectivity, negation, vocabulary, grammar, etc.. but it is also why it is very interesting to working on. 
In this project I choose to try to classify customer feedback from Amazon into “positive” or “negative” sentiment by building a model based on probabilities. The dataset can be obtained from ‘Google datasets’ or ‘Kaggle’. Computers can’t understand text so text has to be converted into a format which is comprehended by the computer


Get the dataset from my github.

## Requirement

We can use multiple programming languages like Python, R, SAS etc.. to classify whether a review is 'Positive' or 'Negative'.But for this project I have used Python since it is ubiquitous, open-source and easy to learn.
Python is a general purpose programming language if you want to know more about python follow this [link](https://en.wikipedia.org/wiki/Python_(programming_language))

And I have used **Jupyter Notebook** to develop the project.Jupyter is a free, open-source, interactive web tool known as a computational notebook, which researchers can use to combine software code, computational output, explanatory text and multimedia resources in a single document.

#### How to Install Python and Jupyter Notebooks  

Installing Python, Jupyter Noteboks and setting the envirnment with required packages or libraries would be cumbersome for a newbie, so I have installed **Anaconda** which would install Python and Jupyter Notebook by default.

##### Step 1  

Download [Anaconda](https://www.anaconda.com/products/individual#windows) for windows

Click on the download button marked by the red arrow.

##### Step 2  
Open the downloaded .exe file 

Click 'Next'  
![Install1](https://user-images.githubusercontent.com/88864828/129460649-e34eb1c5-cb9e-4aa2-ad65-7e34a750f76d.JPG)


Click 'I agree'  for the terms and conditions  
![Install2](https://user-images.githubusercontent.com/88864828/129460663-04573166-e366-4fbd-9010-bcde2126449d.JPG)

Click 'Next' for Just me(recommended) and click on Install
![Install3](https://user-images.githubusercontent.com/88864828/129460669-0d764331-0df4-40dc-9609-bf376b2bcb0b.JPG)

Select the required destination folder where you want to install Anaconda
![Install4](https://user-images.githubusercontent.com/88864828/129460678-3a1a2a60-7724-4374-95c5-313ecfb2bf55.JPG)

Installation has started  
![Install](https://user-images.githubusercontent.com/88864828/129460685-bd7c2f55-1c12-437c-abb7-4f9fd0c6e54d.JPG)

Anconda is Installed Click on Next  
![Install5](https://user-images.githubusercontent.com/88864828/129460768-8bb4f4cb-359d-4958-a620-e2938819e338.jpg)

Click on Next
![Install6](https://user-images.githubusercontent.com/88864828/129460815-ee3700cd-daf9-4df3-9c58-11aea3e9ed9b.JPG)

Click on finish
![Install7](https://user-images.githubusercontent.com/88864828/129460831-5b129aa3-3db4-40ae-9af2-878036b9dc42.JPG)

***Anaconda is successfully Installed!!!***

##### Launch Anaconda Navigator
Open Anaconda Navigator from the Start Menu

![image](https://user-images.githubusercontent.com/88864828/129460868-f97c2d50-5cbb-4c1d-be0e-b2400a37520d.png)

Launch Jupyter Notebooks  

![Install5](https://user-images.githubusercontent.com/88864828/129460983-1f1b350d-6c28-485c-b4f9-3357c392368a.jpg)

Click on 'New' and select  'Python 3'
![image](https://user-images.githubusercontent.com/88864828/129461047-c2dd226c-7fda-49bf-a3a0-936c39430502.png)

Now we can start to use python in Jupyter Notebooks
![image](https://user-images.githubusercontent.com/88864828/129461097-a1ca958e-1213-4605-a660-2906b4ea6d58.png)

Here after we will be using Jupyter Notebooks IDE for the entire project.
## Libraries  
Python has a huge amount of  pre-written librariers which can be leaveraged for this project.

First step we install all the required librariers.

```
# required libraries
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier

from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

#NLTK package is already installed
#But got errors while impelementing tokenization and Lemmetization

#Tokenization
nltk.download('punkt')

#Lemmitization
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```
A quick overview of installed libraries and their use.

Numpy     --> Mathematical Operations  
Pandas    --> Data Analysis  
Matplotlib--> Data Visualization  
Seaborn   --> Data Visualization  
sklearn   --> Machine Learning  
NLTK      --> NLP techniques  
string    --> String Manipluation  
re        --> Regular Expression  


Load the Amazon dataset using pandas library.

Since the dataset is in .csv format we use read_csv function of pandas to load the dataset.

Using head function look at the initial few rows of the dataset,By default head shows the first 5 rows of the dataset
``` 

# load dataset 
df = pd.read_csv('D:\\Trimester 2\\Final Project\\Reviews.csv')

#First 5 rows of the dataset
df.head()
```

![image](https://user-images.githubusercontent.com/88864828/129461394-bcf7c9db-ee37-498f-b7e0-98f5cf5d67ac.png)

Now Let's see the structure of the dataset.
![image](https://user-images.githubusercontent.com/88864828/129461416-0ccef2f0-e994-40b8-8b95-8ff94dbb6eb2.png)


From looking at the structure we can know the following,  
1.There are 10 features/columns and 568454 observations/rows in the dataset.  
2.The datatypes of each feature i.e. int64(for 5 features ), object(for 5 features)  
3.The memory it takes to get stored in the variable(43.4 Mb).  


It is necessary to peek into the data to know what we are deaing with.Sometimes the data is not suitable for analysis so it has to be cleaned.So we need to assess whether our dataset needs to be cleaned or not.  

We remove all the null rows from the dataset. A null denotes no value.  

## Exploratory Data Analysis  

We only keep the features necessary for anlaysis, rest of the features are dropped.So, in our case we only keep 'Text' and 'Score' features and drop the rest.  

![image](https://user-images.githubusercontent.com/88864828/129461781-7985e042-208b-40e8-b122-25d217925adb.png)

We drop null values and then check whether all null values have been removed.  
![image](https://user-images.githubusercontent.com/88864828/129461808-3308bc4c-0b4d-4d11-9b64-52aba5cd25a3.png)


##### Check Random Review

The below code uses random library to generate random numbers and we use this random numbers to look at random rows of the dataset.  
```
#Check random reviews
import random
random.seed(20204749) #Adding seed to reproduce the same result
randomlist = []

#find 5 random numbers within the highest number of rows
for i in range(0,5):
    n = random.randint(1,568454 )
    randomlist.append(n)
print(randomlist)

#Print the reviews and it's corresponding rating
#5 random results are obtained
for i in range(0,5):
    print("Review:","\n", data.loc[randomlist[i]][0]) #print review
    
    print("\n") # create a new line for better readability
    
    print("Rating:","\n", data.loc[randomlist[i]][1]) #print the corresponding rating of above review
    
    print(100*'#') # Adding '#' to sperate one review from other
 ```
![image](https://user-images.githubusercontent.com/88864828/129461866-1b387434-d967-41cd-b8af-7e41f6acdc42.png)

Then we check the Score variable, since score consist of numbers we see the summary statistics of score and the frequency of scores.  
![image](https://user-images.githubusercontent.com/88864828/129461894-6a66b390-7a5b-4817-87f8-cbed07febad5.png)

We can see that the range of scores is between 1 and 5. The scores are skewed towards higher rating/score. If we see the proportion of score 63.87 % contributes to a rating of 5 and 14.18% to rating 4 but only a meagre 21.93% is contributed by ratings 1, 2 and 3 combined. So clearly there is a imbalance in higher and lower rating. This causes any machine learning algorithm to perform bad so we need to make sure our training dataset has balanced reviews to get the best predictive performance. 

###### Distribution of Rating  
```
# distribution of rating
sns.countplot(data['Score'], palette='Blues')

plt.title('Distribution of rating scores')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```

There is a huge imbalance in the data to the high rate classes this may affect the prediction of NLP model.  

Our Aim is to classify review as either positive or negative but there are 5 scores. So we should somehow convert this into a 2 class problem.  
Since this is a 2 class classification problem we convert the response variable accordingly to have only 2 classes. To achieve this we map the scores {1,2,3} to the value 0 and the scores {4,5 } to the value 1. 0 represents a ‘Negative’ comment and 1 represents a ‘Positive’ comment.  
```
# map ratings 1, 2, 3 to 0 (NEGATIVE) and 4, 5 to 1 (POSITIVE) 
sentiment_score = {1: 0,
                   2: 0,
                   3: 0,
                   4: 1,
                   5: 1}

sentiment = {0: 'NEGATIVE',
             1: 'POSITIVE'}


# mapping
data['sentiment_score'] = data['Score'].map(sentiment_score)
data['sentiment'] = data['sentiment_score'].map(sentiment)

data.head()
```

The mapped result looks like below.  
![image](https://user-images.githubusercontent.com/88864828/129461981-f3a7ef44-ac97-4616-8e6a-063bceb70838.png)

##### Visulaize the distribution of positive and negative comments
```
# distribution of sentiment
plt.figure(figsize = (8, 8))

labels = ['POSITIVE', 'NEGATIVE']
colors = ['#189AB4', '#D4F1F4']
plt.pie(data['sentiment'].value_counts(), autopct='%0.2f%%',colors=colors)

plt.title('Distribution of sentiment', size=14, y=-0.01)
plt.legend(labels, ncol=2, loc=9)
plt.show()

```
![image](https://user-images.githubusercontent.com/88864828/129462009-30b56402-0888-4924-9320-c5044f8144a0.png)

##### Visulaization of Negative and Positive Words using Word Cloud  
Word Cloud is a way to visualize textual data, the most repeated words are bigger in size and the less frequent words are smaller in size.

First step we split each word from each individual sentence in all the observations using split function and save the result as a Series.  
```
# get all used words 
all_words = pd.Series(' '.join(data['Text']).split())
```
Then we use the WordCloud function to visulaize the text in our case the reviews.  
```
# plot word cloud
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(all_words))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)
plt.title("Most used words in all reviews", size=16)

plt.axis("off")
plt.show()
```

![image](https://user-images.githubusercontent.com/88864828/129462042-38d81cf2-b317-4857-8293-ecbe5d75311d.png)


Similarly, we split all words only in **positive** reviews.

```
# get words used positive reivews 
positiveWords = pd.Series(' '.join(data[data['sentiment']=='POSITIVE']['Text']).split())
```

 And visualize it using word cloud.  
```
# plot word cloud
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(positiveWords))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)
plt.title("Most used words in positive reviews", size=16)

plt.axis("off")
plt.show()
```
![image](https://user-images.githubusercontent.com/88864828/129462105-0710e267-1132-4b4e-8ee1-ffa04749e35a.png)


Similarly we do for **Negative** words.  

```
# get words used negative reivews 
negativeWords = pd.Series(' '.join(data[data['sentiment']=='NEGATIVE']['Text']).split())

# plot word cloud
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(negativeWords))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)
plt.title("Most used words in negative reviews", size=16)

plt.axis("off")
plt.show()
```
![image](https://user-images.githubusercontent.com/88864828/129462127-f40d3464-24a3-4adb-ad87-3d70405786d6.png)

## Text Processing  
Before doing any analytics it is paramount to check the data quality because the analytics produced from the dataset is as good as the quality of the data. The Amazon dataset has to be cleaned before fitting our machine learning models.  

**1. Clean Text**  
We will go over some simple techniques to clean and prepare text data for modeling with machine learning.
```
def clean_Text(Text:str):
    """ Return cleaned Text:
            - lowercase
            - remove whitespaces
            - remove HTML tags
            - replace digit with spaces
            - replace punctuations with spaces
            - remove extra spaces and tabs
        ------
        input: Text (str)    
        output: cleaned Text (str)
    """
    Text = str(Text)
    
    Text = Text.lower()
    Text = Text.strip()
    
    Text = re.sub(' \d+', ' ', Text)
    Text = re.compile('<.*?>').sub('', Text)
    Text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', Text)
    Text = re.sub('\s+', ' ', Text)
    
    Text = Text.strip()
    
    return Text
 ```  
 We test whether the above clean_Text function can change the text to lowercaase, remove whitespaces, remove html tags, replace digit with spaces and remove extra spaces and tabs.  
 ![image](https://user-images.githubusercontent.com/88864828/129462302-65cef3ed-3f5d-445a-9e04-cd2ac5c1e7f9.png)

**2. Remove Stopwords**  
There can be some words in our sentences that occur very frequently and don't contribute too much to the overall meaning of the sentences. We usually have a list of these words and remove them from each our sentences. For example: "a", "an", "the", "this", "that", "is", "it", "to", "and" in this example.  
```
def remove_stopwords(Text:str):
    """ Remove stopwords from Text:
        ------
        input: Text (str)    
        output: cleaned Text (str)
    """
    Text = str(Text)
    filtered_sentence = []

    # Stop word lists can be adjusted for your problem
    stop_words = ["a", "an", "the", "this", "that", "is", "it", "to", "and"]

    # Tokenize the sentence
    words = word_tokenize(Text)
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    Text = " ".join(filtered_sentence)
    
    return Text
  ```
![image](https://user-images.githubusercontent.com/88864828/129462320-e0d82f70-617a-4c88-8e8f-f075188bda68.png)

Yeah! it works i has removed the stop words from the test string.  

**3. Stemming**  
Stemming is a rule-based system to convert words into their root form. It removes suffixes from words. This helps us enhace similarities (if any) between sentences.

```
def stemm_text(text:str):
    """ Stemm text:
    ------
    input: text (str)    
    output: Stemmed text (str)
    """
    text = str(text)
    # Initialize the stemmer
    snow = SnowballStemmer('english')

    stemmed_sentence = []
    # Tokenize the sentence
    words = word_tokenize(text)
    for w in words:
        # Stem the word/token
        stemmed_sentence.append(snow.stem(w))
    text = " ".join(stemmed_sentence)
    
    return text
 ```
 
 ![image](https://user-images.githubusercontent.com/88864828/129462359-62d30f50-b885-4bca-a26b-9ed467f0b1e7.png)

You can see above that stemming operation is NOT perfect. We have mistakes such as "messag", "involv", "adjac". It is a rule based method that sometimes mistakely remove suffixes from words. Nevertheless, it runs fast.

**4. Lemmatization**  
If we are not satisfied with the result of stemming, we can use the Lemmatization instead. It usually requires more work, but gives better results. As mentioned in the class, lemmatization needs to know the correct word position tags such as "noun", "verb", "adjective", etc. and we will use another NLTK function to feed this information to the lemmatizer.  

The full list of below helper function can be found [here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)  
```
# This is a helper function to map NTLK position tags
# Full list is available here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```

```
def lemmatize(text:str):
    """ lemmatize text:
    ------
    input: text (str)    
    output: lemmatized text (str)
    """
    text = str(text)
    
    # Initialize the lemmatizer
    wl = WordNetLemmatizer()

    lemmatized_sentence = []

    # Tokenize the sentence
    words = word_tokenize(text)
    # Get position tags
    word_pos_tags = nltk.pos_tag(words)
    # Map the position tag and lemmatize the word/token
    for idx, tag in enumerate(word_pos_tags):
        lemmatized_sentence.append(wl.lemmatize(tag[0], get_wordnet_pos(tag[1])))

    lemmatized_text = " ".join(lemmatized_sentence)
    
    return lemmatized_text
```
![image](https://user-images.githubusercontent.com/88864828/129462396-565ffee4-c6eb-48a1-9b47-4237fcff0d2d.png)

This looks better than the stemming result.  

Now we apply the above functions clean_Text, remove_stopwords and lemmatize on our dataset.  

Be aware that the below chunk would take around 2 hours to complete.If we want it to be quick we could opt for stemming rather than lemmatization but it's not accurate as seen from examples above.  

```
# clean text
data['Text'] = data['Text'].apply(clean_Text)
# remove stopwords
data['Text'] = data['Text'].apply(remove_stopwords)
# lemmatize
data['Text'] = data['Text'].apply(lemmatize)
```

## Feature Engineering and Selection of model  
Text data requires special preparation before predictive modelling. The text must be parsed to remove words, called tokenization. Then the words need to be encoded as integers or floating point values as input to a machine learning algorithm, called feature extraction (or vectorization).The scikit-learn library offers easy-to-use tools to perform both tokenization and feature extraction of your text data. 

###### TF-IDF
Using TfidfVectorizer we will tokenize documents, learn the vocabulary and inverse document frequency weightings, and  encode it as a new document. The inverse document frequencies are calculated for each word in the vocabulary, assigning the lowest score of 1.0. The scores are normalized to values between 0 and 1 and the encoded document vectors can then be used directly with most machine learning algorithms.

This is an acronym than stands for “Term Frequency – Inverse Document”

**Term Frequency:** This summarizes how often a given word appears within a document.
**Inverse Document Frequency:** This downscales words that appear a lot across documents.
```
vectorizer = TfidfVectorizer(max_features=700)
vectorizer.fit(data['Text'])
features = vectorizer.transform(data['Text'])

features.toarray()
tf_idf = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names()).astype(np.float16)
# tf_idf.drop('50', axis=1, inplace=True)
tf_idf.head()
```
We can see that now all the explanatory variables(text) are converted into float which is readable by the computer.  

## Split the dataset  
The idea is to train the machine learning model on a specific dataset and test it on unseen data (another dataset). For this purpose we split the given dataset into two sets Training and Testing Sets.  

We would train all our machine learning models on a training datset and Test in testing set to check the predictive performance of the model.  

While splitting we need to make sure that there is no overlap of data between the testing and training set because if there is a overlap the given machine learning model would perform well in the testing set but would fail when deployed in production. 

But this can be easily done with the help of `train_test_split` function wherin we use test_size=0.2( means 20% of data will be allocated for testing dataset and remaining 80% for training), random_state=42(This is a random number given to reproduce the result)  
```
#Random state is similar to seed it helps to reproduce the split
X_train, X_test, y_train, y_test = train_test_split(tf_idf, data['sentiment_score'],  )
```
Now we check the dimension of our splitted train and test dataset

```
print (f'Train set shape\t:{X_train.shape}\nTest set shape\t:{X_test.shape}')
```
![image](https://user-images.githubusercontent.com/88864828/129462726-b79305b7-a96c-44eb-a31e-413a6ed61d21.png)

We can see that there are 700 features in each dataset but initially we had only 10 features, the increase in feature is contributed by TF-IDF

**Result of Split explained**  
X_train --> All features are explanatory variables of training dataset  
X_test  --> All features are explanatory variables of testing dataset  
y_train --> Has only one response variable of training dataset  
y_test  --> Has only one response variable of testing dataset  

## Oversampling  
Imbalanced datasets are those where there is a severe skew in the class distribution.  

This bias in the training dataset can influence many machine learning algorithms, leading some to ignore the minority class entirely. This is a problem as it is typically the minority class on which predictions are most important.  

One approach to addressing the problem of class imbalance is to randomly resample the training dataset. The two main approaches to randomly resampling an imbalanced dataset are to delete examples from the majority class, called undersampling, and to duplicate examples from the minority class, called oversampling.  

Our dataset is skewed.  
![image](https://user-images.githubusercontent.com/88864828/129462883-ff92e320-648e-47c9-828c-2fd160ae8054.png)

So we perform oversampling.  

```
target_count = train_data['sentiment_score'].value_counts()
negative_class = train_data[train_data['sentiment_score'] == 0]
positive_class = train_data[train_data['sentiment_score'] == 1]

negative_over = negative_class.sample(target_count[1], replace=True)

df_train_over = pd.concat([positive_class, negative_over], axis=0)
df_train_over = shuffle(df_train_over)
df_train_over.head()
```

Now we check the frequency of data in each class.  

![image](https://user-images.githubusercontent.com/88864828/129462898-44588476-60d3-4225-98d9-911395548c26.png)

It can be noted that the minority class is '0' with 100011 observations and the major class is '1' with '354752' observations.  

After Oversampling the the count of minority class equals the majority class.

## Models  

Since our data is ready now it's time to fit different models like 

1.
2.
3.
4.
5.
