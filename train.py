import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib


# load csv
music_data = pd.read_csv("music.csv")
# set input
X = music_data.drop(columns=['genre'])
# set output
y = music_data['genre']


# create model object
model = DecisionTreeClassifier()
# train model
model.fit(X,y)

# make a joblib file to get trained model
joblib.dump(model, 'music-recommender.joblib')