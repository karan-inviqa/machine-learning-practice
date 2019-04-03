from sklearn.externals import joblib

# load trained model to predict
model = joblib.load('music-recommender.joblib')

# ask model to predict
predictions = model.predict([[21, 1]])
print(predictions)
