### Text Classifier using Python and Machine Learning from Reddit API

# Imports
import praw
import re
import config
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns




topics = ["machinelearning", "ai", "datascience", "astrology"]

def get_data():

    reddit = praw.Reddit(
            client_id="giS6njwtQgrlAiALZKnO8w",
            client_secret="	_Ad6ysTKvUZ17Ka49npt8-U0ObkcRA",
            password="Excalibur_1",
            user_agent="vitor-app",
            username="AdThis4495"
    )

    char_count = lambda post: len(re.sub('\W|\d', '', post.selftext))

    mask = lambda post: char_count(post) >= 100

    data = []
    labels = []

    for i, topic in enumerate(topics):

        subreddit_Data = reddit.subreddit(topic).new(limit = 1000)

        posts = [post.selftext for post in filter(mask, subreddit_Data)]

        data.extend(posts)
        labels.extend([i], * len(posts))

        print("Number of post from topics: r/ {topic}",
              "\nOne of extracted posts: {posts[0]:[:600]} ...\n",
              "_" + 80 + '\n')

    return data, labels

## Split data in Train and Test

Test_size = 0.2
Random_State = 41

def split_data():
    
    print(f"Split Data {100 * {}}%  for training and validation".format(Test_size))
    
    X_train,X_test, y_train, y_test = train_test_split(data,
                                                       labels,
                                                       test_size= Test_size,
                                                       random_state= Random_State
                                                           )

    return X_train,X_test, y_train, y_test


## Preprocessing

Min_Roc_Freq = 1
N_Components = 1000
N_INTER = 30

def preprocessing():
    pattern = r'\W|\d|http.*\s+|www.*\s+'
    preprocessor = lambda text: re.sub(pattern, ' ', text)
    
    vectorizer = TfidfVectorizer(preprocessor = preprocessor, stop_words='english',  min_df=Min_Roc_Freq)
    
    decomposition = TruncatedSVD(n_components=N_Components, n_iter=N_INTER)
    
    pipeline = [('tfidf', vectorizer), ('svc', decomposition)]
    
    return pipeline

## Model Selection

N_NEIGHBORS = 4
CV = 3

def models_creation():
    
    model1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    model2 = RandomForestClassifier(random_state=Random_State)
    model3 = LogisticRegression(cv = CV, random_state=Random_State)
    
    models = [("KNN", model1), ("Random Forest", model2), ("Logistic Regression", model3)]
    
    return models
