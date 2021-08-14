
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


Similarly, we split all words only in positive reviews.

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


Similarly we do for Negative words.  

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
