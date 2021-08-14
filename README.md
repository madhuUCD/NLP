Sentiment Analysis
This project was motivated by my desire to investigate the sentiment analysis field of machine learning since it makes use of natural language processing (NLP) which is a very hot topic actually. The study of natural language processing has been around for more than 50 years and grew out of the field of linguistics with the rise of computers.
 The Project
Sentiment analysis, also refers as opinion mining, is a sub machine learning task where we want to determine the general sentiment of a given document. Using machine learning techniques and natural language processing we can extract the subjective information of a document and try to classify it according to its polarity such as positive, neutral or negative. It is a really useful analysis since we could possibly determine the overall opinion about selling objects, or predict stock markets for a given company like, if most people think positive about it, possibly its stock markets will increase, and so on. Sentiment analysis is actually far from to be solved since the language is very complex (objectivity/subjectivity, negation, vocabulary, grammar, etc.. but it is also why it is very interesting to working on. 
In this project I choose to try to classify customer feedback from Amazon into “positive” or “negative” sentiment by building a model based on probabilities. The dataset can be obtained from ‘Google datasets’ or ‘Kaggle’. Computers can’t understand text so text has to be converted into a format which is comprehended by the computer
Exploratory Data Analysis (EDA)
We do EDA to get a deeper understanding about the data. The dataset has 10 features/variables and 568,454 observations. Only the necessary features are retained and other features are dropped from the dataset. Since we are dealing with a classification task and the only input variables required is ‘Text’ and ‘Score’ (Figure 1). ‘Text’ has the reviews given by customers and ‘Score’ has the ratings. Then we peak into the reviews present in ‘Text’ feature to know what we are dealing with.

Figure 2 
 
Figure 1  
Distribution of Score
 
We can see that the range of scores is between 1 and 5. The scores are skewed towards higher rating/score. If we see the proportion of score 63.87 % contributes to a rating of 5 and 14.18% to rating 4 but only a meagre 21.93% is contributed by ratings 1, 2 and 3 combined. So clearly there is a imbalance in higher and lower rating. This causes any machine learning algorithm to perform bad so we need to make sure our training dataset has balanced reviews to get the best predictive performance. Since this is a 2 class classification problem we convert the response variable accordingly to have only 2 classes. To achieve this we map the scores {1,2,3} to the value 0 and the scores {4,5 } to the value 1. 0 represents a ‘Negative’ comment and 1 represents a ‘Positive’ comment.
 
Most Used Words
We could see which words were repeated the most in all reviews. The best way to summarize this is to create a word cloud.
 
Positive Reviews 
Negative Reviews
 
Data/Text Pre-processing
Before doing any analytics it is paramount to check the data quality because the analytics produced from the dataset is as good as the quality of the data. The Amazon dataset has to be cleaned before fitting our machine learning models. 
1.	Remove Missing Data and perform following operations
            - lowercase
            - remove whitespaces
            - remove HTML tags
            - replace digit with spaces
            - replace punctuations with spaces
            - remove extra spaces and tabs

2.	Remove Stop Words
Stop words are the words which despite removed from a sentence would make sense. Stop words in sentences occur very frequently and don't contribute too much to the overall meaning of the sentences. We usually have a list of these words and remove them from each our sentences. For example: "a", "an", "the", "this", "that", "is", "it", "to", "and".
3.	Stemming / Lemmatization   
Stemming and Lemmatizing is the process of reducing a word to its root form. The main purpose is to reduce variations of the same word, thereby reducing the corpus of words we include in the model. The difference between stemming and lemmatizing is that, stemming chops off the end of the word without taking into consideration the context of the word. Whereas, Lemmatizing considers the context of the word and shortens the word into its root form based on the dictionary definition. Stemming is a faster process compared to Lemmantization. Hence,  it a trade-off between speed and accuracy.
Tokenization
Text data requires special preparation before predictive modelling. The text must be parsed to remove words, called tokenization. Then the words need to be encoded as integers or floating point values as input to a machine learning algorithm, called feature extraction (or vectorization).The scikit-learn library offers easy-to-use tools to perform both tokenization and feature extraction of your text data. 
Using TfidfVectorizer we will tokenize documents, learn the vocabulary and inverse document frequency weightings, and  encode it as a new document. The inverse document frequencies are calculated for each word in the vocabulary, assigning the lowest score of 1.0. The scores are normalized to values between 0 and 1 and the encoded document vectors can then be used directly with most machine learning algorithms.

Splitting the dataset
We split the dataset into training and testing sets. The training set with a proportion of 80% and the testing set with a proportion of 20%.
Model Building
The final step in the text classification framework is to train a classifier using the features created in the previous step. There are many different choices of machine learning models which can be used to train a final model. We will implement following different classifiers for this purpose
1. Naive Bayes Classifier
2. Linear Classifier
3. Support Vector Machine
4. Bagging Models
5. Boosting Models
6. Shallow Neural Networks
7. Deep Neural Networks
8. Convolutional Neural Network (CNN)

