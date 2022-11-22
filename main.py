### Text Classifier using Python and Machine Learning from Reddit API

# Imports
import praw
import re
import numpy as np
import config
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


input_string = input("Enter a list element separated by space ")
input_string
topics  = input_string.split()

 #= ["machinelearning", "datascience", "astrology"]

def get_data():

    reddit = praw.Reddit(
            client_id="giS6njwtQgrlAiALZKnO8w",
            client_secret="_Ad6ysTKvUZ17Ka49npt8-U0ObkcRA",
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
        labels.extend([i] * len(posts))
        
        print(f"Número de posts do assunto r/{topics}: {len(posts)}",
              f"\nUm dos posts extraídos: {posts[0][:600]}...\n",
              "_" * 80 + '\n')

        

    return data, labels

## Split data in Train and Test

Test_size = 0.2
Random_State = 0

def split_data():

    print(f"Split {100 * Test_size}% for training and validation...")

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
    model3 = LogisticRegressionCV(cv = CV, random_state=Random_State, max_iter = 3000)

    models = [("KNN", model1), ("Random Forest", model2), ("Logistic Regression", model3)]

    return models


## Train Model

def train_model(models, pipeline, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models:

        pipe = Pipeline(pipeline + [(name, model)])

        print(f"Training model {name} with train data...")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        report = classification_report(y_test, y_pred)
        print("Classifier Report: {y_pred} \n")

        results.append([model, {'model': name, 'prediction': y_pred, 'report': report}])

    return results


## Run pipeline
if __name__ == "__main__":

    data, labels = get_data()

    X_train,X_test, y_train, y_test = split_data()

    pipeline = preprocessing()

    models = models_creation()

    results = train_model(models, pipeline, X_train,X_test, y_train, y_test)

print("Finished")

def plot_distribution():
    _, counts = np.unique(labels, return_counts = True)
    sns.set_theme(style = "whitegrid")
    plt.figure(figsize = (15, 6), dpi = 120)
    plt.title("Number of post per Topics")
    sns.barplot(x = topics, y = counts)
    plt.legend([' '.join([f.title(),f"- {c} posts"]) for f,c in zip(topics, counts)])
    plt.show()

def plot_confusion(result):
    print("Classification Report\n", result[-1]['report'])
    y_pred = result[-1]['prediction']
    conf_matrix = confusion_matrix(y_test, y_pred)
    _, test_counts = np.unique(y_test, return_counts = True)
    conf_matrix_percent = conf_matrix / test_counts.transpose() * 100
    plt.figure(figsize = (9,8), dpi = 120)
    plt.title(result[-1]['model'].upper() + " Results")
    plt.xlabel("Real Value")
    plt.ylabel("Model Prediction")
    ticklabels = [f"r/{sub}" for sub in topics]
    sns.heatmap(data = conf_matrix_percent, xticklabels = ticklabels, yticklabels = ticklabels, annot = True, fmt = '.2f')
    plt.show()


# Gráfico de avaliação
plot_distribution()

# Resultado do KNN
plot_confusion(results[0])

# Resultado do RandomForest
plot_confusion(results[1])

# Resultado da Regressão Logística
plot_confusion(results[2])


# Fim