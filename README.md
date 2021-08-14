
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


